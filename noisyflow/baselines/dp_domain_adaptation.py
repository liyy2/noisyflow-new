from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.nn import MLP
from noisyflow.stage3.training import eval_classifier
from noisyflow.utils import DPConfig, unwrap_model


def _make_private_with_mode(
    privacy_engine,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp: DPConfig,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader]:
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


def _train_dp_erm_classifier_impl(
    train_loader: DataLoader,
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
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu",
    name: str = "Baseline/DP-ERM",
) -> Tuple[nn.Module, Dict[str, float]]:
    if hidden is None:
        hidden = [256, 256]
    model = MLP(int(d), int(num_classes), hidden=list(hidden), act="silu").to(device)
    if init_state_dict is not None:
        model.load_state_dict(copy.deepcopy(init_state_dict))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    privacy_engine = None
    if dp is not None and dp.enabled:
        try:
            from opacus import PrivacyEngine
        except Exception as exc:
            raise RuntimeError("Opacus not installed but DPConfig.enabled=True. Install opacus or disable DP.") from exc
        try:
            privacy_engine = PrivacyEngine(secure_mode=getattr(dp, "secure_mode", False))
        except TypeError:
            privacy_engine = PrivacyEngine()
        model, opt, train_loader = _make_private_with_mode(privacy_engine, model, opt, train_loader, dp)

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
        if ep % max(1, int(epochs) // 5) == 0:
            stats = eval_classifier(_AsClassifier(model), test_loader, device=device)
            print(f"[{name}] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}  test_acc={stats['acc']:.3f}")
            model.train()

    model_eval: nn.Module = model
    if ref_finetune_loader is not None and int(ref_finetune_epochs) > 0:
        # Detach from Opacus GradSampleModule hooks by copying weights into a fresh model.
        base = unwrap_model(model)
        clean = MLP(int(d), int(num_classes), hidden=list(hidden), act="silu").to(device)
        clean.load_state_dict(base.state_dict())
        model_eval = clean

        ft_lr = float(lr if ref_finetune_lr is None else ref_finetune_lr)
        ft_opt = torch.optim.Adam(model_eval.parameters(), lr=ft_lr)
        model_eval.train()
        for ep in range(1, int(ref_finetune_epochs) + 1):
            for xb, yb in ref_finetune_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                logits = model_eval(xb)
                loss = F.cross_entropy(logits, yb)
                ft_opt.zero_grad(set_to_none=True)
                loss.backward()
                ft_opt.step()
                last_loss = float(loss.detach().cpu().item())
            if ep % max(1, int(ref_finetune_epochs) // 2) == 0:
                stats = eval_classifier(_AsClassifier(model_eval), test_loader, device=device)
                print(
                    f"[{name}/finetune] epoch {ep:04d}/{ref_finetune_epochs}  loss={last_loss:.4f}  test_acc={stats['acc']:.3f}"
                )
                model_eval.train()

    out: Dict[str, float] = {"clf_loss": float(last_loss)}
    out.update(eval_classifier(_AsClassifier(model_eval), test_loader, device=device))
    if privacy_engine is not None and dp is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon"] = eps
        out["delta"] = float(dp.delta)
    return unwrap_model(model_eval), out


def train_dp_erm_classifier(
    train_loader: DataLoader,
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
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu",
    name: str = "Baseline/DP-ERM",
) -> Dict[str, float]:
    """
    Train a DP (or non-DP) MLP classifier with standard ERM.

    Args:
        train_loader: DataLoader over labeled training data (x, y).
        test_loader: DataLoader over labeled evaluation data (x, y).
        d: Feature dimension.
        num_classes: Number of classes.
        hidden: Hidden layer sizes for the MLP.
        epochs: Training epochs.
        lr: Learning rate.
        dp: Optional DPConfig. If provided and enabled, uses Opacus DP-SGD.
        ref_finetune_loader: Optional labeled DataLoader used for post-processing fine-tuning
            (e.g., public target reference labels). This does not change the DP budget.
        ref_finetune_epochs: Number of post-processing fine-tuning epochs on ref_finetune_loader.
        ref_finetune_lr: Optional learning rate for fine-tuning (defaults to lr).
        init_state_dict: Optional model initialization used before local training. This is
            useful for federated parameter averaging baselines where every client must start
            from the same weights.
        device: Torch device string.
        name: Label used in logging.

    Returns:
        Dict[str, float] with keys including 'acc' and (if DP) 'epsilon'/'delta'.
    """
    _, out = _train_dp_erm_classifier_impl(
        train_loader,
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
        init_state_dict=init_state_dict,
        device=device,
        name=name,
    )
    return out


def train_dp_erm_classifier_with_model(
    train_loader: DataLoader,
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
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu",
    name: str = "Baseline/DP-ERM",
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train a DP (or non-DP) ERM classifier and also return the trained model.
    """
    return _train_dp_erm_classifier_impl(
        train_loader,
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
        init_state_dict=init_state_dict,
        device=device,
        name=name,
    )


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:  # type: ignore[override]
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.coeff * grad_output, None


def _grl(x: torch.Tensor, coeff: float) -> torch.Tensor:
    return _GradReverse.apply(x, float(coeff))


@dataclass(frozen=True)
class DANNConfig:
    feature_hidden: List[int]
    feature_dim: int
    label_hidden: List[int]
    domain_hidden: List[int]
    lambda_domain: float


class _DANN(nn.Module):
    def __init__(
        self,
        *,
        d: int,
        num_classes: int,
        cfg: DANNConfig,
    ) -> None:
        super().__init__()
        self.feature_extractor = MLP(int(d), int(cfg.feature_dim), hidden=list(cfg.feature_hidden), act="silu")
        self.label_head = MLP(int(cfg.feature_dim), int(num_classes), hidden=list(cfg.label_hidden), act="silu")
        self.domain_head = MLP(int(cfg.feature_dim), 2, hidden=list(cfg.domain_hidden), act="silu")
        self.lambda_domain = float(cfg.lambda_domain)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(x)
        class_logits = self.label_head(feats)
        domain_logits = self.domain_head(_grl(feats, self.lambda_domain))
        return class_logits, domain_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.label_head(feats)


class _AsClassifier(nn.Module):
    """
    Adapter: treat an arbitrary module as a stage3.Classifier-compatible model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = unwrap_model(self.model)
        if hasattr(inner, "predict"):
            return inner.predict(x)  # type: ignore[no-any-return]
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def _train_dp_dann_impl(
    train_ds: TensorDataset,
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    batch_size: int = 256,
    device: str = "cpu",
    cfg: Optional[DANNConfig] = None,
    name: str = "Baseline/DP-DANN",
) -> Tuple[nn.Module, Dict[str, float]]:
    if cfg is None:
        cfg = DANNConfig(
            feature_hidden=[256],
            feature_dim=128,
            label_hidden=[],
            domain_hidden=[128],
            lambda_domain=1.0,
        )
    model = _DANN(d=int(d), num_classes=int(num_classes), cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    privacy_engine = None
    if dp is not None and dp.enabled:
        try:
            from opacus import PrivacyEngine
        except Exception as exc:
            raise RuntimeError("Opacus not installed but DPConfig.enabled=True. Install opacus or disable DP.") from exc
        try:
            privacy_engine = PrivacyEngine(secure_mode=getattr(dp, "secure_mode", False))
        except TypeError:
            privacy_engine = PrivacyEngine()
        model, opt, loader = _make_private_with_mode(privacy_engine, model, opt, loader, dp)

    last_loss = float("nan")
    model.train()
    for ep in range(1, int(epochs) + 1):
        for xb, y_class, y_domain in loader:
            xb = xb.to(device).float()
            y_class = y_class.to(device).long().view(-1)
            y_domain = y_domain.to(device).long().view(-1)

            class_logits, domain_logits = model(xb)
            domain_loss = F.cross_entropy(domain_logits, y_domain)

            labeled_mask = y_class >= 0
            if bool(labeled_mask.any().item()):
                class_loss = F.cross_entropy(class_logits[labeled_mask], y_class[labeled_mask])
            else:
                class_loss = torch.zeros((), device=device)

            loss = class_loss + domain_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

        if ep % max(1, int(epochs) // 5) == 0:
            stats = eval_classifier(_AsClassifier(model), test_loader, device=device)
            print(f"[{name}] epoch {ep:04d}/{epochs}  loss={last_loss:.4f}  test_acc={stats['acc']:.3f}")
            model.train()

    out: Dict[str, float] = {"clf_loss": float(last_loss)}
    out.update(eval_classifier(_AsClassifier(model), test_loader, device=device))
    if privacy_engine is not None and dp is not None:
        eps = float(privacy_engine.get_epsilon(delta=dp.delta))
        out["epsilon"] = eps
        out["delta"] = float(dp.delta)
    return unwrap_model(model), out


def train_dp_dann(
    train_ds: TensorDataset,
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    batch_size: int = 256,
    device: str = "cpu",
    cfg: Optional[DANNConfig] = None,
    name: str = "Baseline/DP-DANN",
) -> Dict[str, float]:
    """
    Domain-adversarial training (DANN) on a combined dataset.

    The training dataset must be a TensorDataset with tensors:
      - x: (N,d) float
      - y_class: (N,) int64, with -1 for unlabeled target samples
      - y_domain: (N,) int64 in {0 (source), 1 (target)}

    DP (if enabled) is applied to the entire training dataset via Opacus DP-SGD.

    Returns:
        Dict[str, float] with keys including 'acc' and (if DP) 'epsilon'/'delta'.
    """
    _, out = _train_dp_dann_impl(
        train_ds,
        test_loader,
        d=d,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        dp=dp,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        name=name,
    )
    return out


def train_dp_dann_with_model(
    train_ds: TensorDataset,
    test_loader: DataLoader,
    *,
    d: int,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
    dp: Optional[DPConfig] = None,
    batch_size: int = 256,
    device: str = "cpu",
    cfg: Optional[DANNConfig] = None,
    name: str = "Baseline/DP-DANN",
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train a DP (or non-DP) DANN model and also return the trained model.
    """
    return _train_dp_dann_impl(
        train_ds,
        test_loader,
        d=d,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        dp=dp,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        name=name,
    )
