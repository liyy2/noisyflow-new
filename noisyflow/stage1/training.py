from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from noisyflow.stage1.networks import ConditionalVAE, VelocityField
from noisyflow.utils import DPConfig, unwrap_model


def _make_private_with_mode(
    privacy_engine,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp: DPConfig,
):
    grad_sample_mode = getattr(dp, "grad_sample_mode", None)
    if grad_sample_mode is not None:
        try:
            return privacy_engine.make_private(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=dp.noise_multiplier,
                max_grad_norm=dp.max_grad_norm,
                grad_sample_mode=grad_sample_mode,
            )
        except TypeError:
            pass
    return privacy_engine.make_private(
        module=module,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp.noise_multiplier,
        max_grad_norm=dp.max_grad_norm,
    )


def _build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer = str(optimizer).lower()
    weight_decay = float(weight_decay)
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0")
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError("optimizer must be one of {'adam','adamw','sgd'}")


def _configure_privacy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    *,
    epochs: int,
    dp: Optional[DPConfig],
):
    privacy_engine = None
    if dp is None or not dp.enabled:
        return model, optimizer, loader, privacy_engine

    try:
        from opacus import PrivacyEngine
    except Exception as e:
        raise RuntimeError(
            "Opacus not installed but DPConfig.enabled=True. Install opacus or disable DP."
        ) from e

    try:
        privacy_engine = PrivacyEngine(secure_mode=getattr(dp, "secure_mode", False))
    except TypeError:
        privacy_engine = PrivacyEngine()

    target_epsilon = getattr(dp, "target_epsilon", None)
    if target_epsilon is not None:
        if not hasattr(privacy_engine, "make_private_with_epsilon"):
            raise RuntimeError(
                "DPConfig.target_epsilon requested but this Opacus version does not support "
                "PrivacyEngine.make_private_with_epsilon. Upgrade opacus or set dp.noise_multiplier."
            )
        grad_sample_mode = getattr(dp, "grad_sample_mode", None)
        kwargs = {
            "module": model,
            "optimizer": optimizer,
            "data_loader": loader,
            "target_epsilon": float(target_epsilon),
            "target_delta": float(dp.delta),
            "epochs": int(epochs),
            "max_grad_norm": float(dp.max_grad_norm),
        }
        if grad_sample_mode is not None:
            kwargs["grad_sample_mode"] = grad_sample_mode
        try:
            model, optimizer, loader = privacy_engine.make_private_with_epsilon(**kwargs)
        except (TypeError, ValueError):
            kwargs.pop("grad_sample_mode", None)
            model, optimizer, loader = privacy_engine.make_private_with_epsilon(**kwargs)
    else:
        model, optimizer, loader = _make_private_with_mode(privacy_engine, model, optimizer, loader, dp)

    return model, optimizer, loader, privacy_engine


class _EMA:
    def __init__(self, params: Tuple[torch.nn.Parameter, ...], decay: float) -> None:
        self.decay = float(decay)
        if not (0.0 < self.decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        self.params = tuple(p for p in params if p.requires_grad)
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        for shadow, param in zip(self.shadow, self.params, strict=True):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self) -> None:
        for shadow, param in zip(self.shadow, self.params, strict=True):
            param.data.copy_(shadow)


def _batch_memory_context(
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    privacy_engine,
    dp: Optional[DPConfig],
):
    if privacy_engine is None or dp is None:
        return nullcontext(loader)
    max_physical_batch_size = getattr(dp, "max_physical_batch_size", None)
    if max_physical_batch_size is None:
        return nullcontext(loader)
    try:
        from opacus.utils.batch_memory_manager import BatchMemoryManager
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "dp.max_physical_batch_size requires opacus.utils.batch_memory_manager (upgrade opacus)."
        ) from exc
    return BatchMemoryManager(
        data_loader=loader,
        max_physical_batch_size=int(max_physical_batch_size),
        optimizer=optimizer,
    )


def _unpack_stage1_batch(
    batch,
    *,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)):
        xb = batch[0]
        yb = batch[1] if len(batch) >= 2 else None
        cond = batch[2] if len(batch) >= 3 else None
    else:
        xb = batch
        yb = None
        cond = None

    if yb is None:
        raise ValueError("Stage I requires labels; expected DataLoader to yield (x, y[, cond]).")

    xb = xb.to(device).float()
    yb = yb.to(device).long()
    cond_t = cond.to(device).float() if cond is not None else None
    return xb, yb, cond_t


def _scalarize_metrics(metrics: Dict[str, torch.Tensor | float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)
    return out


def _train_stage1_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    ema_decay: Optional[float],
    dp: Optional[DPConfig],
    device: str,
    loss_fn: Callable[
        [torch.nn.Module, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor | float]],
    ],
) -> Dict[str, float]:
    model.to(device)
    model.train()

    opt = _build_optimizer(model, optimizer=optimizer, lr=lr, weight_decay=weight_decay)
    model, opt, loader, privacy_engine = _configure_privacy(model, opt, loader, epochs=epochs, dp=dp)

    ema: Optional[_EMA] = None
    if ema_decay is not None:
        ema = _EMA(tuple(model.parameters()), decay=float(ema_decay))

    last_metrics: Dict[str, float] = {}
    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        with _batch_memory_context(loader, opt, privacy_engine=privacy_engine, dp=dp) as epoch_loader:
            for batch in epoch_loader:
                xb, yb, cond_t = _unpack_stage1_batch(batch, device=device)
                loss, metrics = loss_fn(model, xb, yb, cond_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update()
                last_loss = float(loss.detach().cpu().item())
                last_metrics = _scalarize_metrics(metrics)

        if ep % max(1, epochs // 5) == 0:
            print(f"[Stage I] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}")

    if ema is not None:
        ema.apply()

    out: Dict[str, float] = dict(last_metrics)
    out.setdefault("stage1_loss", last_loss)
    if privacy_engine is not None and dp is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon_stage1"] = eps
        out["delta_stage1"] = float(dp.delta)
        nm = getattr(privacy_engine, "noise_multiplier", None)
        if nm is not None:
            out["noise_multiplier_stage1"] = float(nm)
        print(f"[Stage I] DP eps={eps:.3f}, delta={dp.delta:g}")
    return out


def flow_matching_loss(
    f: VelocityField,
    x: torch.Tensor,
    y: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    normalize_by_dim: bool = False,
) -> torch.Tensor:
    """
    Sample z ~ N(0,I), t ~ Unif[0,1],
      x_t = (1-t) z + t x
      v*  = x - z
    Minimize || f(x_t, t, y) - v* ||^2
    """
    z = torch.randn_like(x)
    t = torch.rand(x.shape[0], 1, device=x.device)
    x_t = (1.0 - t) * z + t * x
    v_star = x - z
    v = f(x_t, t, y, cond=cond)
    sq = (v - v_star) ** 2
    if normalize_by_dim:
        return sq.mean(dim=1).mean()
    return sq.sum(dim=1).mean()


def vae_loss(
    model: ConditionalVAE,
    x: torch.Tensor,
    y: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    *,
    beta: float = 1.0,
    normalize_by_dim: bool = False,
    return_parts: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon, mu, logvar = model(x, y, cond=cond)
    model_base = unwrap_model(model)
    recon_sq = (recon - x) ** 2
    if normalize_by_dim:
        recon_per_example = recon_sq.mean(dim=1)
    else:
        recon_per_example = recon_sq.sum(dim=1)

    kl_per_example = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    if normalize_by_dim:
        kl_per_example = kl_per_example / float(max(1, model_base.latent_dim))

    recon_term = recon_per_example.mean()
    kl_term = kl_per_example.mean()
    total = recon_term + float(beta) * kl_term
    if return_parts:
        return total, recon_term, kl_term
    return total


@torch.no_grad()
def sample_flow_euler(
    f: VelocityField,
    labels: torch.Tensor,
    n_steps: int = 50,
    z0: Optional[torch.Tensor] = None,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solves z' = f(z,t,label), t in [0,1], Euler discretization.
    Returns z_1.

    labels: (B,) int64
    z0: (B,d) optional, else sample N(0,I)
    """
    device = labels.device
    steps = max(1, int(n_steps))
    z = torch.randn(labels.shape[0], f.d, device=device) if z0 is None else z0.to(device)
    dt = 1.0 / float(steps)
    for k in range(steps):
        t = torch.full((labels.shape[0], 1), float(k) / float(steps), device=device)
        z = z + dt * f(z, t, labels, cond=cond)
    return z


@torch.no_grad()
def sample_vae(
    model: ConditionalVAE,
    labels: torch.Tensor,
    z0: Optional[torch.Tensor] = None,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = labels.device
    z = (
        torch.randn(labels.shape[0], model.latent_dim, device=device)
        if z0 is None
        else z0.to(device=device, dtype=torch.float32)
    )
    return model.decode(z, labels, cond=cond)


def train_flow_stage1(
    f: VelocityField,
    loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    ema_decay: Optional[float] = None,
    loss_normalize_by_dim: bool = False,
    dp: Optional[DPConfig] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Client-side training of a flow-matching Stage I model.
    """

    def compute_loss(
        model: torch.nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        cond_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = flow_matching_loss(
            model,
            xb,
            yb,
            cond=cond_t,
            normalize_by_dim=bool(loss_normalize_by_dim),
        )
        return loss, {"stage1_loss": loss, "flow_loss": loss}

    out = _train_stage1_model(
        f,
        loader,
        epochs=epochs,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        dp=dp,
        device=device,
        loss_fn=compute_loss,
    )
    if "epsilon_stage1" in out:
        out["epsilon_flow"] = float(out["epsilon_stage1"])
        out["delta_flow"] = float(out["delta_stage1"])
        if "noise_multiplier_stage1" in out:
            out["noise_multiplier_flow"] = float(out["noise_multiplier_stage1"])
    return out


def train_vae_stage1(
    model: ConditionalVAE,
    loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    ema_decay: Optional[float] = None,
    loss_normalize_by_dim: bool = False,
    beta: float = 1.0,
    dp: Optional[DPConfig] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Client-side training of a class-conditional Stage I VAE.
    """

    def compute_loss(
        vae: torch.nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        cond_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total, recon, kl = vae_loss(
            vae,
            xb,
            yb,
            cond=cond_t,
            beta=float(beta),
            normalize_by_dim=bool(loss_normalize_by_dim),
            return_parts=True,
        )
        return total, {
            "stage1_loss": total,
            "vae_loss": total,
            "vae_recon_loss": recon,
            "vae_kl_loss": kl,
        }

    return _train_stage1_model(
        model,
        loader,
        epochs=epochs,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        dp=dp,
        device=device,
        loss_fn=compute_loss,
    )
