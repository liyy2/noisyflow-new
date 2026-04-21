from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import train_classifier, train_random_forest_classifier


@dataclass(frozen=True)
class IJCai2019DPOTConfig:
    projection_dim: int = 30
    target_ot_size: Optional[int] = 5000
    source_ot_size: Optional[int] = None
    sinkhorn_reg: float = 30.0
    sinkhorn_iters: int = 200
    sinkhorn_eps: float = 1e-9
    labelwise_ot: bool = True
    noise_ratio: float = 0.3
    delta: float = 1e-5
    label_epsilon: Optional[float] = None
    source_clip_norm: Optional[float] = None
    classifier: str = "auto"  # auto | rf | mlp
    device: str = "cpu"
    seed: int = 0


def epsilon_from_noise_ratio(noise_ratio: float, delta: float) -> float:
    """
    Convert the DPOT noise-ratio r = sigma / w into an (epsilon, delta) guarantee.

    From Theorem 2 (IJCAI'19 DPOT), sigma >= w * sqrt(2(ln(1/(2δ)) + ε)) / ε.
    Solving for ε using r = sigma/w yields:
        r^2 ε^2 - 2 ε - 2 ln(1/(2δ)) = 0
        ε = (1 + sqrt(1 + 2 r^2 ln(1/(2δ)))) / r^2
    """
    r = float(noise_ratio)
    if r <= 0.0:
        raise ValueError("noise_ratio must be > 0")
    delta = float(delta)
    if not (0.0 < delta < 0.5):
        raise ValueError("delta must be in (0, 0.5)")
    ln_term = math.log(1.0 / (2.0 * delta))
    return float((1.0 + math.sqrt(1.0 + 2.0 * (r * r) * ln_term)) / (r * r))


def _clip_rows_l2(x: torch.Tensor, clip_norm: float) -> torch.Tensor:
    clip_norm = float(clip_norm)
    if clip_norm <= 0.0:
        raise ValueError("clip_norm must be > 0")
    x = x.to(dtype=torch.float64)
    norms = x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(clip_norm / norms, max=1.0)
    return x * scale


def _noisy_labels_from_histogram(
    y_true_sorted: torch.Tensor,
    *,
    num_classes: int,
    epsilon: float,
    seed: int,
) -> torch.Tensor:
    """
    DPDA-style label transfer: reorder by true labels, send noisy histogram counts, and reconstruct a
    length-N label vector from the noisy counts.
    """
    if int(y_true_sorted.numel()) == 0:
        raise ValueError("y_true_sorted is empty")
    num_classes = int(num_classes)
    if num_classes <= 1:
        raise ValueError("num_classes must be >= 2")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("label epsilon must be > 0")

    y_np = y_true_sorted.detach().cpu().numpy().astype(np.int64, copy=False)
    counts = np.bincount(y_np, minlength=num_classes).astype(np.float64, copy=False)

    rng = np.random.default_rng(int(seed))
    noisy = counts + rng.laplace(loc=0.0, scale=1.0 / eps, size=num_classes)
    noisy = np.maximum(0.0, np.rint(noisy)).astype(np.int64, copy=False)

    n = int(y_np.shape[0])
    total = int(noisy.sum())
    if total > n:
        # Reduce counts starting from the largest bins.
        excess = total - n
        order = np.argsort(-noisy)
        for idx in order.tolist():
            if excess <= 0:
                break
            take = min(excess, int(noisy[idx]))
            noisy[idx] -= take
            excess -= take
    elif total < n:
        # Assign remaining labels to the current majority class.
        deficit = n - total
        if int(noisy.sum()) == 0:
            noisy[0] = n
        else:
            noisy[int(noisy.argmax())] += deficit

    labels = np.empty((n,), dtype=np.int64)
    pos = 0
    for c in range(num_classes):
        k = int(noisy[c])
        if k <= 0:
            continue
        labels[pos : pos + k] = c
        pos += k
        if pos >= n:
            break
    if pos < n:
        labels[pos:] = int(labels[pos - 1]) if pos > 0 else 0

    return torch.from_numpy(labels).to(dtype=torch.long)


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1).unsqueeze(0)
    return x2 + y2 - 2.0 * (x @ y.t())


def _sinkhorn_scaling(
    cost: torch.Tensor,
    *,
    reg: float,
    iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute entropic OT scaling factors (u, v) for uniform marginals using Sinkhorn-Knopp.

    Returns (u, v, K) where K = exp(-cost/reg).
    """
    if cost.dim() != 2:
        raise ValueError("cost must be a 2D matrix")
    n_s, n_t = int(cost.shape[0]), int(cost.shape[1])
    if n_s == 0 or n_t == 0:
        raise ValueError("Empty cost matrix")

    reg = float(reg)
    if reg <= 0.0:
        raise ValueError("reg must be > 0")

    cost = cost.to(dtype=torch.float64)
    cost = cost - cost.min()

    K = torch.exp(-cost / reg).to(dtype=torch.float64)
    a = torch.full((n_s,), 1.0 / float(n_s), dtype=torch.float64, device=K.device)
    b = torch.full((n_t,), 1.0 / float(n_t), dtype=torch.float64, device=K.device)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(iters)):
        Kv = K @ v
        u = a / (Kv + float(eps))
        Ktu = K.t() @ u
        v = b / (Ktu + float(eps))

    return u, v, K


@torch.no_grad()
def dpot_barycentric_transport(
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    *,
    projection_dim: int,
    noise_ratio: float,
    source_clip_norm: Optional[float],
    sinkhorn_reg: float,
    sinkhorn_iters: int,
    sinkhorn_eps: float,
    seed: int,
    device: str,
    source_y: Optional[torch.Tensor] = None,
    target_y: Optional[torch.Tensor] = None,
    labelwise_ot: bool = True,
) -> torch.Tensor:
    """
    IJCAI'19 DPOT-style transport via noisy JL projection + entropic OT + barycentric mapping.

    If labelwise_ot=True and (source_y, target_y) are provided, solves OT per class using the same
    DP-protected projected source data (post-processing).
    """
    if source_x.dim() != 2 or target_x.dim() != 2:
        raise ValueError("source_x and target_x must be 2D (N,d)")
    if int(source_x.shape[1]) != int(target_x.shape[1]):
        raise ValueError("source_x and target_x must share the same feature dimension")

    n_s = int(source_x.shape[0])
    if n_s == 0:
        raise ValueError("Empty source_x")
    d = int(source_x.shape[1])
    l = int(projection_dim)
    if l <= 0:
        raise ValueError("projection_dim must be > 0")

    torch_device = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    M = (torch.randn((d, l), generator=gen, dtype=torch.float64) / math.sqrt(float(l))).to(device=torch_device)
    w = M.pow(2).sum(dim=1).sqrt().max().item()
    clip_norm = 1.0 if source_clip_norm is None else float(source_clip_norm)
    sigma = float(noise_ratio) * float(w) * float(clip_norm)

    source_in = source_x
    if source_clip_norm is not None:
        source_in = _clip_rows_l2(source_x, float(source_clip_norm))
    source_proj = source_in.to(dtype=torch.float64) @ M
    noise = torch.randn(source_proj.shape, generator=gen, dtype=torch.float64).to(device=torch_device) * float(sigma)
    source_noisy = source_proj + noise
    target_proj = target_x.to(dtype=torch.float64) @ M

    bias = float(l) * (float(sigma) ** 2)

    if labelwise_ot and source_y is not None and target_y is not None:
        source_y = source_y.to(device="cpu").long().view(-1)
        target_y = target_y.to(device="cpu").long().view(-1)
        if int(source_y.numel()) != int(source_x.shape[0]):
            raise ValueError("source_y must align with source_x")
        if int(target_y.numel()) != int(target_x.shape[0]):
            raise ValueError("target_y must align with target_x")

        out = source_x.to(dtype=torch.float64).clone()
        for c in torch.unique(source_y).tolist():
            c_int = int(c)
            src_idx = (source_y == c_int).nonzero(as_tuple=False).view(-1).to(device=torch_device)
            tgt_idx = (target_y == c_int).nonzero(as_tuple=False).view(-1).to(device=torch_device)
            if int(src_idx.numel()) == 0 or int(tgt_idx.numel()) == 0:
                continue

            xs = source_noisy.index_select(0, src_idx)
            yt = target_proj.index_select(0, tgt_idx)
            cost = _pairwise_sq_dists(xs, yt) - bias
            u, v, K = _sinkhorn_scaling(
                cost,
                reg=sinkhorn_reg,
                iters=sinkhorn_iters,
                eps=sinkhorn_eps,
            )
            # gamma = u[:,None] * K * v[None,:] ; barycentric map: ns * gamma @ X_t
            weighted = (K * v.unsqueeze(0)) @ target_x.index_select(0, tgt_idx).to(dtype=torch.float64)
            trans = float(src_idx.numel()) * (u.unsqueeze(1) * weighted)
            out.index_copy_(0, src_idx, trans)
        return out.to(device=device, dtype=torch.float32)

    cost = _pairwise_sq_dists(source_noisy, target_proj) - bias
    u, v, K = _sinkhorn_scaling(cost, reg=sinkhorn_reg, iters=sinkhorn_iters, eps=sinkhorn_eps)
    weighted = (K * v.unsqueeze(0)) @ target_x.to(dtype=torch.float64)
    trans = float(n_s) * (u.unsqueeze(1) * weighted)
    return trans.to(device=device, dtype=torch.float32)


def _subsample_labeled_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    n: Optional[int],
    *,
    num_classes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n is None:
        return x, y
    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")
    if n >= int(x.shape[0]):
        return x, y
    y_np = y.detach().cpu().numpy().astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    per_class = max(1, n // max(1, int(num_classes)))
    idx: List[int] = []
    for c in range(int(num_classes)):
        idx_c = np.flatnonzero(y_np == c)
        if idx_c.size == 0:
            continue
        rng.shuffle(idx_c)
        idx.extend(idx_c[: min(per_class, idx_c.size)].tolist())
    if len(idx) < n:
        all_idx = np.arange(y_np.shape[0])
        mask = np.ones(y_np.shape[0], dtype=bool)
        mask[np.array(idx, dtype=np.int64)] = False
        remaining = all_idx[mask]
        rng.shuffle(remaining)
        idx.extend(remaining[: (n - len(idx))].tolist())
    idx_np = np.array(idx[:n], dtype=np.int64)
    rng.shuffle(idx_np)
    idx_t = torch.from_numpy(idx_np).to(device=x.device)
    return x.index_select(0, idx_t), y.index_select(0, idx_t)


def run_ijcai2019_dpot_experiment(
    *,
    client_datasets: List[TensorDataset],
    target_ref: TensorDataset,
    target_test: TensorDataset,
    num_classes: int,
    cfg: IJCai2019DPOTConfig,
    ref_train_size: int,
    combined_train_size: Optional[int],
    batch_size: int = 512,
) -> Dict[str, float]:
    device = str(cfg.device)
    x_ref = target_ref.tensors[0].to(device).float()
    y_ref = target_ref.tensors[1].to(device).long().view(-1)
    x_test = target_test.tensors[0].to(device).float()
    y_test = target_test.tensors[1].to(device).long().view(-1)

    x_ref_ot, y_ref_ot = _subsample_labeled_tensors(
        x_ref,
        y_ref,
        cfg.target_ot_size,
        num_classes=num_classes,
        seed=cfg.seed,
    )

    transported_x: List[torch.Tensor] = []
    transported_y: List[torch.Tensor] = []
    for i, ds in enumerate(client_datasets):
        xs = ds.tensors[0].to(device).float()
        ys = ds.tensors[1].to(device).long().view(-1)
        if cfg.source_ot_size is not None:
            xs, ys = _subsample_labeled_tensors(
                xs,
                ys,
                cfg.source_ot_size,
                num_classes=num_classes,
                seed=cfg.seed + 17 * i,
            )
        # DPDA-style label transfer: reorder by true labels and reconstruct noisy labels from a noisy histogram.
        order = torch.argsort(ys)
        xs = xs.index_select(0, order)
        ys_sorted = ys.index_select(0, order)
        if cfg.label_epsilon is not None:
            ys_priv = _noisy_labels_from_histogram(
                ys_sorted,
                num_classes=num_classes,
                epsilon=float(cfg.label_epsilon),
                seed=int(cfg.seed) + 997 * i,
            ).to(device=device)
        else:
            ys_priv = ys_sorted
        xt = dpot_barycentric_transport(
            xs,
            x_ref_ot,
            projection_dim=cfg.projection_dim,
            noise_ratio=cfg.noise_ratio,
            source_clip_norm=cfg.source_clip_norm,
            sinkhorn_reg=cfg.sinkhorn_reg,
            sinkhorn_iters=cfg.sinkhorn_iters,
            sinkhorn_eps=cfg.sinkhorn_eps,
            seed=cfg.seed + 31 * i,
            device=device,
            source_y=ys_priv,
            target_y=y_ref_ot,
            labelwise_ot=cfg.labelwise_ot,
        )
        transported_x.append(xt.detach().cpu())
        transported_y.append(ys_priv.detach().cpu())

    x_tr = torch.cat(transported_x, dim=0)
    y_tr = torch.cat(transported_y, dim=0)

    # Train loaders.
    x_ref_train, y_ref_train = _subsample_labeled_tensors(
        x_ref.detach().cpu(),
        y_ref.detach().cpu(),
        ref_train_size,
        num_classes=num_classes,
        seed=cfg.seed,
    )
    x_tr_train, y_tr_train = _subsample_labeled_tensors(
        x_tr,
        y_tr,
        combined_train_size,
        num_classes=num_classes,
        seed=cfg.seed + 1,
    )

    ref_loader = DataLoader(
        TensorDataset(x_ref_train, y_ref_train),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    combo_loader = DataLoader(
        TensorDataset(torch.cat([x_ref_train, x_tr_train], dim=0), torch.cat([y_ref_train, y_tr_train], dim=0)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(TensorDataset(x_test.detach().cpu(), y_test.detach().cpu()), batch_size=int(batch_size))

    out: Dict[str, float] = {}
    choice = str(cfg.classifier).strip().lower()
    if choice not in {"auto", "rf", "random_forest", "mlp"}:
        raise ValueError(f"cfg.classifier must be one of: auto, rf, mlp (got '{cfg.classifier}')")
    require_rf = choice in {"rf", "random_forest"}
    use_rf = choice in {"auto", "rf", "random_forest"}

    if use_rf:
        try:
            ref_stats = train_random_forest_classifier(
                ref_loader,
                test_loader=test_loader,
                seed=cfg.seed,
                name="Classifier/RF-ref_only (DPOT)",
            )
        except RuntimeError:
            if require_rf:
                raise
            ref_clf = Classifier(d=int(x_ref.shape[1]), num_classes=num_classes, hidden=[128, 128])
            ref_stats = train_classifier(
                ref_clf,
                ref_loader,
                test_loader=test_loader,
                epochs=20,
                lr=1e-3,
                device=device,
            )
    else:
        ref_clf = Classifier(d=int(x_ref.shape[1]), num_classes=num_classes, hidden=[128, 128])
        ref_stats = train_classifier(ref_clf, ref_loader, test_loader=test_loader, epochs=20, lr=1e-3, device=device)
    out["acc_ref_only"] = float(ref_stats.get("acc", float("nan")))

    if use_rf:
        try:
            combo_stats = train_random_forest_classifier(
                combo_loader,
                test_loader=test_loader,
                seed=cfg.seed,
                name="Classifier/RF-ref+transport (DPOT)",
            )
        except RuntimeError:
            if require_rf:
                raise
            combo_clf = Classifier(d=int(x_ref.shape[1]), num_classes=num_classes, hidden=[128, 128])
            combo_stats = train_classifier(
                combo_clf,
                combo_loader,
                test_loader=test_loader,
                epochs=20,
                lr=1e-3,
                device=device,
            )
    else:
        combo_clf = Classifier(d=int(x_ref.shape[1]), num_classes=num_classes, hidden=[128, 128])
        combo_stats = train_classifier(
            combo_clf,
            combo_loader,
            test_loader=test_loader,
            epochs=20,
            lr=1e-3,
            device=device,
        )
    out["acc_ref_plus_transport"] = float(combo_stats.get("acc", float("nan")))

    eps_ot = epsilon_from_noise_ratio(cfg.noise_ratio, cfg.delta)
    eps_total = eps_ot
    if cfg.label_epsilon is not None:
        eps_total += float(cfg.label_epsilon)
    out["epsilon_total"] = float(eps_total)
    out["epsilon_ot"] = float(eps_ot)
    out["delta"] = float(cfg.delta)
    return out
