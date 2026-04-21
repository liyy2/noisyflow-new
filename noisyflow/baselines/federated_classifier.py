from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from noisyflow.baselines.dp_domain_adaptation import train_dp_erm_classifier_with_model
from noisyflow.nn import MLP
from noisyflow.stage3.training import eval_classifier
from noisyflow.utils import DPConfig, unwrap_model


def average_model_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Average compatible model state dicts.

    Floating-point tensors are averaged with optional weights. Non-floating tensors are copied
    from the first state dict unchanged.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    if weights is None:
        weights = [1.0] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise ValueError("weights must have the same length as state_dicts")

    total_weight = float(sum(float(w) for w in weights))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value")

    keys = list(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        if list(state_dict.keys()) != keys:
            raise ValueError("all state_dicts must share the same keys")

    averaged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key in keys:
        tensors = [state_dict[key].detach().cpu() for state_dict in state_dicts]
        template = tensors[0]
        if torch.is_floating_point(template):
            accum = torch.zeros_like(template, dtype=torch.float64)
            for weight, tensor in zip(weights, tensors):
                accum.add_(tensor.to(dtype=torch.float64), alpha=float(weight))
            averaged[key] = (accum / total_weight).to(dtype=template.dtype)
        else:
            averaged[key] = template.clone()
    return averaged


def _finetune_supervised(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str,
    name: str,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    model = copy.deepcopy(unwrap_model(model)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    last_loss = float("nan")
    model.train()
    for ep in range(1, int(epochs) + 1):
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
        if ep % max(1, int(epochs) // 2) == 0:
            stats = eval_classifier(model, test_loader, device=device)
            print(f"[{name}] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}  test_acc={stats['acc']:.3f}")
            model.train()

    out: Dict[str, float] = {"clf_loss": float(last_loss)}
    out.update(eval_classifier(model, test_loader, device=device))
    return model, out


def train_fedavg_classifier_with_model(
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    hidden: Optional[List[int]] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    ref_finetune_loader: Optional[DataLoader] = None,
    ref_finetune_epochs: int = 0,
    ref_finetune_lr: Optional[float] = None,
    device: str = "cpu",
    name: str = "Baseline/FedAvg",
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Train one classifier per client from a shared initialization, then average parameters."""
    if not client_loaders:
        raise ValueError("client_loaders must be non-empty")

    if hidden is None:
        hidden = [256, 256]

    init_model = MLP(int(d), int(num_classes), hidden=list(hidden), act="silu")
    init_state_dict = copy.deepcopy(init_model.state_dict())

    local_models: List[torch.nn.Module] = []
    local_stats: List[Dict[str, float]] = []
    local_weights: List[float] = []
    epsilons: List[float] = []

    for client_idx, client_loader in enumerate(client_loaders):
        dataset = getattr(client_loader, "dataset", None)
        if dataset is None:
            raise ValueError("each client_loader must expose a dataset for FedAvg weighting")
        local_weights.append(float(len(dataset)))
        model_i, stats_i = train_dp_erm_classifier_with_model(
            client_loader,
            test_loader,
            d=d,
            num_classes=num_classes,
            hidden=list(hidden),
            epochs=int(epochs),
            lr=float(lr),
            dp=dp,
            init_state_dict=init_state_dict,
            device=device,
            name=f"{name}-client{client_idx}",
        )
        local_models.append(unwrap_model(model_i))
        local_stats.append(stats_i)
        if "epsilon" in stats_i:
            epsilons.append(float(stats_i["epsilon"]))

    averaged_state = average_model_state_dicts(
        [copy.deepcopy(model.state_dict()) for model in local_models],
        weights=local_weights,
    )
    global_model = MLP(int(d), int(num_classes), hidden=list(hidden), act="silu").to(device)
    global_model.load_state_dict(averaged_state)

    out: Dict[str, float] = {"clf_loss": float("nan")}
    out.update(eval_classifier(global_model, test_loader, device=device))
    out["client_count"] = float(len(local_models))
    out["client_weight_total"] = float(sum(local_weights))
    out["client_acc_mean"] = float(
        sum(float(stats.get("acc", 0.0)) for stats in local_stats) / max(1, len(local_stats))
    )
    if epsilons:
        out["epsilon_max"] = float(max(epsilons))
        out["epsilon_mean"] = float(sum(epsilons) / len(epsilons))
        if dp is not None:
            out["delta"] = float(dp.delta)

    if ref_finetune_loader is not None and int(ref_finetune_epochs) > 0:
        global_model, finetune_stats = _finetune_supervised(
            global_model,
            ref_finetune_loader,
            test_loader,
            epochs=int(ref_finetune_epochs),
            lr=float(lr if ref_finetune_lr is None else ref_finetune_lr),
            device=device,
            name=f"{name}/finetune",
        )
        out.update(finetune_stats)

    return unwrap_model(global_model), out


def train_fedavg_classifier(
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    hidden: Optional[List[int]] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    ref_finetune_loader: Optional[DataLoader] = None,
    ref_finetune_epochs: int = 0,
    ref_finetune_lr: Optional[float] = None,
    device: str = "cpu",
    name: str = "Baseline/FedAvg",
) -> Dict[str, float]:
    _, stats = train_fedavg_classifier_with_model(
        client_loaders,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        dp=dp,
        ref_finetune_loader=ref_finetune_loader,
        ref_finetune_epochs=ref_finetune_epochs,
        ref_finetune_lr=ref_finetune_lr,
        device=device,
        name=name,
    )
    return stats
