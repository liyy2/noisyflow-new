from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from noisyflow.nn import MLP
from noisyflow.stage3.training import eval_classifier


def _resolve_betas(beta: Union[float, Sequence[float]], n_clients: int) -> List[float]:
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    if isinstance(beta, Sequence) and not isinstance(beta, (str, bytes)):
        betas = [float(b) for b in beta]
        if len(betas) != n_clients:
            raise ValueError("beta sequence must have one entry per source client")
    else:
        betas = [float(beta)] * n_clients
    for b in betas:
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"beta must lie in [0, 1], got {b}")
    return betas


def _positive_projection(target_update: torch.Tensor, source_update: torch.Tensor) -> torch.Tensor:
    denom = torch.dot(source_update, source_update)
    if float(denom.item()) <= 0.0:
        return torch.zeros_like(target_update)
    coeff = torch.dot(target_update, source_update) / denom
    coeff = torch.clamp(coeff, min=0.0)
    return coeff * source_update


def _train_local_model(
    init_state_dict: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    hidden: Sequence[int],
    epochs: int,
    lr: float,
    device: str,
) -> Tuple[torch.nn.Module, float]:
    model = MLP(int(d), int(num_classes), hidden=list(hidden), act="silu").to(device)
    model.load_state_dict(copy.deepcopy(init_state_dict))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    last_loss = float("nan")
    model.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
    return model, last_loss


def train_fedgp_classifier_with_model(
    client_loaders: Sequence[DataLoader],
    target_loader: DataLoader,
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    hidden: Optional[Sequence[int]] = None,
    rounds: int = 25,
    source_epochs: int = 1,
    target_epochs: int = 1,
    lr: float = 1e-3,
    server_lr: float = 1.0,
    beta: Union[float, Sequence[float]] = 0.5,
    device: str = "cpu",
    name: str = "Baseline/FedGP",
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Train a predictor-only FedGP baseline.

    This implements the fixed-beta FedGP aggregation rule from Jiang et al. (ICLR 2024):

      (1 / N) * sum_i [ (1 - beta_i) * g_T + beta_i * Proj_+(g_T | g_Si) ]

    where g_T is the target-client local update and g_Si is the i-th source-client local update.
    """
    if not client_loaders:
        raise ValueError("client_loaders must be non-empty")
    if hidden is None:
        hidden = [256, 256]
    if float(server_lr) <= 0.0:
        raise ValueError("server_lr must be positive")

    betas = _resolve_betas(beta, len(client_loaders))
    global_model = MLP(int(d), int(num_classes), hidden=list(hidden), act="silu").to(device)

    last_target_loss = float("nan")
    last_source_loss_mean = float("nan")
    last_positive_fraction = float("nan")

    for round_idx in range(1, int(rounds) + 1):
        base_state = copy.deepcopy(global_model.state_dict())
        base_vec = parameters_to_vector(global_model.parameters()).detach().clone()

        target_model, last_target_loss = _train_local_model(
            base_state,
            target_loader,
            d=d,
            num_classes=num_classes,
            hidden=list(hidden),
            epochs=int(target_epochs),
            lr=float(lr),
            device=device,
        )
        target_vec = parameters_to_vector(target_model.parameters()).detach().clone()
        target_update = target_vec - base_vec

        aggregated_update = torch.zeros_like(base_vec)
        source_losses: List[float] = []
        positive_count = 0
        for beta_i, client_loader in zip(betas, client_loaders):
            source_model, source_loss = _train_local_model(
                base_state,
                client_loader,
                d=d,
                num_classes=num_classes,
                hidden=list(hidden),
                epochs=int(source_epochs),
                lr=float(lr),
                device=device,
            )
            source_vec = parameters_to_vector(source_model.parameters()).detach().clone()
            source_update = source_vec - base_vec
            proj_update = _positive_projection(target_update, source_update)
            if float(torch.dot(target_update, source_update).item()) > 0.0:
                positive_count += 1
            aggregated_update.add_((1.0 - beta_i) * target_update + beta_i * proj_update)
            source_losses.append(float(source_loss))

        aggregated_update.div_(float(len(client_loaders)))
        new_vec = base_vec + float(server_lr) * aggregated_update
        vector_to_parameters(new_vec, global_model.parameters())

        last_source_loss_mean = float(sum(source_losses) / max(1, len(source_losses)))
        last_positive_fraction = float(positive_count / max(1, len(client_loaders)))

        if round_idx % max(1, int(rounds) // 5) == 0:
            stats = eval_classifier(global_model, test_loader, device=device)
            print(
                f"[{name}] round {round_idx:04d}/{rounds}  "
                f"target_loss={last_target_loss:.4f}  source_loss={last_source_loss_mean:.4f}  "
                f"pos_frac={last_positive_fraction:.2f}  test_acc={stats['acc']:.3f}"
            )

    out: Dict[str, float] = {
        "beta_mean": float(sum(betas) / len(betas)),
        "rounds": float(rounds),
        "source_epochs": float(source_epochs),
        "target_epochs": float(target_epochs),
        "server_lr": float(server_lr),
        "target_loss": float(last_target_loss),
        "source_loss_mean": float(last_source_loss_mean),
        "positive_projection_fraction": float(last_positive_fraction),
    }
    out.update(eval_classifier(global_model, test_loader, device=device))
    return global_model, out


def train_fedgp_classifier(
    client_loaders: Sequence[DataLoader],
    target_loader: DataLoader,
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    hidden: Optional[Sequence[int]] = None,
    rounds: int = 25,
    source_epochs: int = 1,
    target_epochs: int = 1,
    lr: float = 1e-3,
    server_lr: float = 1.0,
    beta: Union[float, Sequence[float]] = 0.5,
    device: str = "cpu",
    name: str = "Baseline/FedGP",
) -> Dict[str, float]:
    _, stats = train_fedgp_classifier_with_model(
        client_loaders,
        target_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        rounds=rounds,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        lr=lr,
        server_lr=server_lr,
        beta=beta,
        device=device,
        name=name,
    )
    return stats
