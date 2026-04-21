from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from noisyflow.metrics import sliced_w2_distance
from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
from noisyflow.stage2.training import train_ot_stage2, train_ot_stage2_cellot, train_ot_stage2_rectified_flow
from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import train_classifier, train_random_forest_classifier
from noisyflow.utils import set_seed, unwrap_model


@dataclass(frozen=True)
class NoiseThenOTConfig:
    """Config for the 'noise data then learn OT' baseline.

    This baseline removes the DP generator (Stage 1). Each client sanitizes the original data
    using per-example L2 clipping + Gaussian noise, trains an OT map on the sanitized data,
    then releases transported samples. OT training and transport are post-processing.
    """

    target_epsilon: float
    delta: float = 1e-5
    clip_norm: float = 1.0
    seed: int = 0


def gaussian_noise_multiplier_for_epsilon(target_epsilon: float, delta: float) -> float:
    """Return the Gaussian noise multiplier that achieves `target_epsilon` (classic bound).

    Uses the standard (non-tight) Gaussian mechanism calibration:
        sigma >= S * sqrt(2 ln(1.25/delta)) / epsilon
    where S is the L2 sensitivity. With per-example L2 clipping to `clip_norm` and releasing
    a vector of all sanitized examples, S = 2 * clip_norm. If we parameterize sigma as
        sigma = noise_multiplier * clip_norm
    then clip_norm cancels and:
        noise_multiplier = 2 * sqrt(2 ln(1.25/delta)) / epsilon
    """
    eps = float(target_epsilon)
    if eps <= 0.0:
        raise ValueError("target_epsilon must be > 0")
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return float(2.0 * math.sqrt(2.0 * math.log(1.25 / delta)) / eps)


def epsilon_from_gaussian_noise_multiplier(noise_multiplier: float, delta: float) -> float:
    """Inverse of `gaussian_noise_multiplier_for_epsilon`."""
    nm = float(noise_multiplier)
    if nm <= 0.0:
        raise ValueError("noise_multiplier must be > 0")
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return float(2.0 * math.sqrt(2.0 * math.log(1.25 / delta)) / nm)


def _clip_rows_l2(x: torch.Tensor, clip_norm: float) -> torch.Tensor:
    clip_norm = float(clip_norm)
    if clip_norm <= 0.0:
        raise ValueError("clip_norm must be > 0")
    x = x.to(dtype=torch.float32)
    norms = x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(clip_norm / norms, max=1.0)
    return x * scale


def make_noised_dataset(
    ds: TensorDataset,
    *,
    noise_multiplier: float,
    clip_norm: float,
    seed: int,
) -> TensorDataset:
    """Return a sanitized dataset (x_noised, y) by clipping and adding Gaussian noise."""
    if not isinstance(ds, TensorDataset) or len(ds.tensors) < 2:
        raise ValueError("Expected a labeled TensorDataset with tensors (x, y, ...)")
    x = ds.tensors[0].detach().cpu().to(dtype=torch.float32)
    y = ds.tensors[1].detach().cpu().to(dtype=torch.long).view(-1)
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("Dataset tensors (x, y) must have matching first dimension")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    x_clip = _clip_rows_l2(x, clip_norm=float(clip_norm))
    sigma = float(noise_multiplier) * float(clip_norm)
    noise = torch.randn(x_clip.shape, generator=gen, dtype=torch.float32) * sigma
    x_noised = x_clip + noise
    return TensorDataset(x_noised, y)


def _sample_rows(
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_total = int(x.shape[0])
    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")
    if n_total <= 0:
        raise ValueError("Empty dataset")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if n <= n_total:
        idx = torch.randperm(n_total, generator=gen)[:n]
    else:
        idx = torch.randint(0, n_total, (n,), generator=gen)
    return x.index_select(0, idx), y.index_select(0, idx)


def _infer_dims(
    client_datasets: List[TensorDataset],
    *,
    target_ref: Optional[TensorDataset] = None,
    target_test: Optional[TensorDataset] = None,
) -> Tuple[int, int]:
    d = int(client_datasets[0].tensors[0].shape[1])
    label_tensors: List[torch.Tensor] = []
    for ds in client_datasets:
        if isinstance(ds, TensorDataset) and len(ds.tensors) >= 2:
            label_tensors.append(ds.tensors[1])
    for ds in (target_ref, target_test):
        if isinstance(ds, TensorDataset) and len(ds.tensors) >= 2:
            label_tensors.append(ds.tensors[1])
    max_label = -1
    for t in label_tensors:
        if t.numel() == 0:
            continue
        max_label = max(max_label, int(t.max().item()))
    if max_label < 0:
        raise ValueError("Could not infer num_classes (no non-empty label tensors found).")
    return d, int(max_label + 1)


def _subsample_labeled_dataset(ds: TensorDataset, n: Optional[int], num_classes: int, seed: int) -> TensorDataset:
    if n is None:
        return ds
    n = int(n)
    if n <= 0:
        raise ValueError("ref_train_size must be > 0")
    if n >= len(ds):
        return ds
    labels = ds.tensors[1].long().cpu().numpy()
    rng = np.random.default_rng(seed)

    per_class = max(1, n // max(1, num_classes))
    indices: List[int] = []
    for c in range(num_classes):
        idx_c = np.flatnonzero(labels == c)
        if idx_c.size == 0:
            continue
        rng.shuffle(idx_c)
        indices.extend(idx_c[: min(per_class, idx_c.size)].tolist())

    if len(indices) < n:
        all_idx = np.arange(labels.shape[0])
        mask = np.ones(labels.shape[0], dtype=bool)
        mask[np.array(indices, dtype=np.int64)] = False
        remaining = all_idx[mask]
        rng.shuffle(remaining)
        indices.extend(remaining[: (n - len(indices))].tolist())

    idx = np.array(indices[:n], dtype=np.int64)
    rng.shuffle(idx)
    tensors = [t[idx] for t in ds.tensors]
    return TensorDataset(*tensors)


def _transport(ot: torch.nn.Module, x: torch.Tensor, *, device: str) -> torch.Tensor:
    if isinstance(ot, RectifiedFlowOT):
        return ot.transport(x.to(device).float()).detach().cpu()
    with torch.enable_grad():
        x_req = x.to(device).float().detach().requires_grad_(True)
        y = ot.transport(x_req)
    return y.detach().cpu()


def run_noise_then_ot_experiment(
    *,
    client_datasets: List[TensorDataset],
    target_ref: TensorDataset,
    target_test: TensorDataset,
    cfg,
    noise_cfg: NoiseThenOTConfig,
) -> Dict[str, float]:
    """Run the noise-then-OT baseline for a single epsilon value.

    Args:
        client_datasets: Per-client private datasets.
        target_ref: Target-domain reference dataset (public).
        target_test: Target-domain test dataset.
        cfg: ExperimentConfig-like object (uses .device, .loaders, .stage2, .stage3, .seed).
        noise_cfg: NoiseThenOTConfig describing the privacy mechanism.

    Returns:
        Stats dictionary mirroring keys produced by `run.py` for comparability.
    """
    device = str(getattr(cfg, "device", "cpu"))
    seed = int(getattr(cfg, "seed", 0))
    set_seed(seed)
    d, num_classes = _infer_dims(client_datasets, target_ref=target_ref, target_test=target_test)

    noise_multiplier = gaussian_noise_multiplier_for_epsilon(noise_cfg.target_epsilon, noise_cfg.delta)
    epsilon = epsilon_from_gaussian_noise_multiplier(noise_multiplier, noise_cfg.delta)

    target_loader = DataLoader(
        target_ref,
        batch_size=int(cfg.loaders.target_batch_size),
        shuffle=True,
        drop_last=False,
    )
    target_test_loader = DataLoader(
        target_test,
        batch_size=int(cfg.loaders.test_batch_size),
        shuffle=False,
    )

    xs_raw: List[torch.Tensor] = []
    ys_transported: List[torch.Tensor] = []
    ls: List[torch.Tensor] = []

    for idx, ds in enumerate(client_datasets):
        noised_ds = make_noised_dataset(
            ds,
            noise_multiplier=noise_multiplier,
            clip_norm=noise_cfg.clip_norm,
            seed=int(noise_cfg.seed) + idx,
        )
        source_loader = DataLoader(
            noised_ds,
            batch_size=int(cfg.loaders.batch_size),
            shuffle=True,
            drop_last=bool(cfg.loaders.drop_last),
        )

        use_cellot = bool(cfg.stage2.cellot.enabled)
        use_rectified_flow = bool(cfg.stage2.rectified_flow.enabled)
        if use_cellot and use_rectified_flow:
            raise ValueError("Choose only one Stage2 model: stage2.cellot.enabled or stage2.rectified_flow.enabled.")

        if use_cellot:
            if str(cfg.stage2.option).upper() != "A":
                raise ValueError("CellOT mode currently supports stage2.option A only.")
            kernel_init = None
            try:
                from run import _kernel_init_from_config  # Local import for consistency with run.py.

                kernel_init = _kernel_init_from_config(dict(cfg.stage2.cellot.kernel_init))
            except Exception:
                kernel_init = None
            f = CellOTICNN(
                input_dim=d,
                hidden_units=list(cfg.stage2.cellot.hidden_units),
                activation=str(cfg.stage2.cellot.activation),
                softplus_W_kernels=bool(cfg.stage2.cellot.softplus_W_kernels),
                softplus_beta=float(cfg.stage2.cellot.softplus_beta),
                fnorm_penalty=float(cfg.stage2.cellot.f_fnorm_penalty),
                kernel_init_fxn=kernel_init,
            )
            ot = CellOTICNN(
                input_dim=d,
                hidden_units=list(cfg.stage2.cellot.hidden_units),
                activation=str(cfg.stage2.cellot.activation),
                softplus_W_kernels=bool(cfg.stage2.cellot.softplus_W_kernels),
                softplus_beta=float(cfg.stage2.cellot.softplus_beta),
                fnorm_penalty=float(cfg.stage2.cellot.g_fnorm_penalty),
                kernel_init_fxn=kernel_init,
            )
            train_ot_stage2_cellot(
                f,
                ot,
                source_loader=source_loader,
                target_loader=target_loader,
                epochs=int(cfg.stage2.epochs),
                n_inner_iters=int(cfg.stage2.cellot.n_inner_iters),
                lr_f=float(cfg.stage2.lr),
                lr_g=float(cfg.stage2.lr),
                optim_cfg=dict(cfg.stage2.cellot.optim),
                n_iters=cfg.stage2.cellot.n_iters,
                dp=None,
                synth_sampler=None,
                device=device,
            )
        elif use_rectified_flow:
            ot = RectifiedFlowOT(
                d=d,
                hidden=list(cfg.stage2.rectified_flow.hidden),
                time_emb_dim=int(cfg.stage2.rectified_flow.time_emb_dim),
                act=str(cfg.stage2.rectified_flow.act),
                transport_steps=int(cfg.stage2.rectified_flow.transport_steps),
                mlp_norm=str(cfg.stage2.rectified_flow.mlp_norm),
                mlp_dropout=float(cfg.stage2.rectified_flow.mlp_dropout),
            )
            train_ot_stage2_rectified_flow(
                ot,
                source_loader=source_loader,
                target_loader=target_loader,
                option="A",
                pair_by_label=bool(cfg.stage2.pair_by_label),
                pair_by_ot=bool(cfg.stage2.pair_by_ot),
                pair_by_ot_method=str(cfg.stage2.pair_by_ot_method),
                synth_sampler=None,
                epochs=int(cfg.stage2.epochs),
                lr=float(cfg.stage2.lr),
                optimizer=str(cfg.stage2.optimizer),
                weight_decay=float(cfg.stage2.weight_decay),
                ema_decay=cfg.stage2.ema_decay,
                loss_normalize_by_dim=bool(cfg.stage2.loss_normalize_by_dim),
                public_synth_steps=int(cfg.stage2.public_synth_steps),
                public_pretrain_epochs=int(cfg.stage2.public_pretrain_epochs),
                dp=None,
                device=device,
            )
        else:
            ot = ICNN(
                d=d,
                hidden=list(cfg.stage2.hidden),
                act=str(cfg.stage2.act),
                add_strong_convexity=float(cfg.stage2.add_strong_convexity),
            )
            train_ot_stage2(
                ot,
                real_loader=source_loader,
                target_loader=target_loader,
                option="A",
                synth_sampler=None,
                epochs=int(cfg.stage2.epochs),
                lr=float(cfg.stage2.lr),
                dp=None,
                conj_steps=int(cfg.stage2.conj_steps),
                conj_lr=float(cfg.stage2.conj_lr),
                conj_clamp=cfg.stage2.conj_clamp,
                device=device,
            )

        ot_model = unwrap_model(ot).to(device).eval()
        x_noised = noised_ds.tensors[0]
        y_noised = noised_ds.tensors[1]
        x_raw_i, labels_i = _sample_rows(
            x_noised,
            y_noised,
            int(cfg.stage3.M_per_client),
            seed=seed + 10_000 + idx,
        )
        y_trans_i = _transport(ot_model, x_raw_i, device=device)
        xs_raw.append(x_raw_i.cpu())
        ys_transported.append(y_trans_i.cpu())
        ls.append(labels_i.cpu())

    x_syn_raw = torch.cat(xs_raw, dim=0)
    y_syn = torch.cat(ys_transported, dim=0)
    l_syn = torch.cat(ls, dim=0)

    stats_sw2: Dict[str, float] = {}
    try:
        x_ref = target_ref.tensors[0]
        x_private = torch.cat([ds.tensors[0] for ds in client_datasets], dim=0)
        stats_sw2["sw2_private_ref"] = float(
            sliced_w2_distance(x_private, x_ref, num_projections=128, max_samples=2000, seed=seed)
        )
        stats_sw2["sw2_synth_ref"] = float(
            sliced_w2_distance(x_syn_raw, x_ref, num_projections=128, max_samples=2000, seed=seed)
        )
        stats_sw2["sw2_synth_transported_ref"] = float(
            sliced_w2_distance(y_syn, x_ref, num_projections=128, max_samples=2000, seed=seed)
        )
    except Exception:
        stats_sw2 = {}

    syn_loader = DataLoader(
        TensorDataset(y_syn, l_syn),
        batch_size=int(cfg.loaders.synth_batch_size),
        shuffle=True,
        drop_last=bool(cfg.loaders.drop_last),
    )

    try:
        stats = train_random_forest_classifier(
            syn_loader,
            test_loader=target_test_loader,
            seed=seed,
            name="Classifier/RF-synth",
        )
        clf = None
    except RuntimeError:
        clf = Classifier(d=d, num_classes=num_classes, hidden=list(cfg.stage3.hidden))
        stats = train_classifier(
            clf,
            syn_loader,
            test_loader=target_test_loader,
            epochs=int(cfg.stage3.epochs),
            lr=float(cfg.stage3.lr),
            device=device,
        )

    out: Dict[str, float] = dict(stats)
    out["epsilon_noise"] = float(epsilon)
    out["delta_noise"] = float(noise_cfg.delta)
    out["noise_multiplier_noise"] = float(noise_multiplier)
    out["clip_norm_noise"] = float(noise_cfg.clip_norm)

    syn_raw_loader = DataLoader(
        TensorDataset(x_syn_raw, l_syn),
        batch_size=int(cfg.loaders.synth_batch_size),
        shuffle=True,
        drop_last=bool(cfg.loaders.drop_last),
    )
    try:
        raw_stats = train_random_forest_classifier(
            syn_raw_loader,
            test_loader=target_test_loader,
            seed=seed,
            name="Classifier/RF-synth_raw",
        )
    except RuntimeError:
        raw_clf = Classifier(d=d, num_classes=num_classes, hidden=list(cfg.stage3.hidden))
        raw_stats = train_classifier(
            raw_clf,
            syn_raw_loader,
            test_loader=target_test_loader,
            epochs=int(cfg.stage3.epochs),
            lr=float(cfg.stage3.lr),
            device=device,
        )
    out["acc_syn_raw"] = float(raw_stats.get("acc", float("nan")))
    out["f1_syn_raw"] = float(raw_stats.get("f1_macro", float("nan")))
    out.update(stats_sw2)

    nan = float("nan")
    out.setdefault("clf_loss_ref_only", nan)
    out.setdefault("acc_ref_only", nan)
    out.setdefault("f1_ref_only", nan)
    out.setdefault("clf_loss_ref_plus_synth", nan)
    out.setdefault("acc_ref_plus_synth", nan)
    out.setdefault("f1_ref_plus_synth", nan)

    if isinstance(target_ref, TensorDataset) and len(target_ref.tensors) >= 2:
        ref_supervised_ds = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
        ref_supervised_ds = _subsample_labeled_dataset(
            ref_supervised_ds,
            n=cfg.stage3.ref_train_size,
            num_classes=num_classes,
            seed=seed,
        )
        ref_train_loader = DataLoader(
            ref_supervised_ds,
            batch_size=int(cfg.loaders.target_batch_size),
            shuffle=True,
            drop_last=bool(cfg.loaders.drop_last),
        )
        try:
            ref_stats = train_random_forest_classifier(
                ref_train_loader,
                test_loader=target_test_loader,
                seed=seed,
                name="Classifier/RF-ref_only",
            )
        except RuntimeError:
            ref_clf = Classifier(d=d, num_classes=num_classes, hidden=list(cfg.stage3.hidden))
            ref_stats = train_classifier(
                ref_clf,
                ref_train_loader,
                test_loader=target_test_loader,
                epochs=int(cfg.stage3.epochs),
                lr=float(cfg.stage3.lr),
                device=device,
            )
        out["clf_loss_ref_only"] = float(ref_stats.get("clf_loss", nan))
        out["acc_ref_only"] = float(ref_stats.get("acc", nan))
        out["f1_ref_only"] = float(ref_stats.get("f1_macro", nan))

        syn_supervised_ds = TensorDataset(y_syn, l_syn)
        syn_supervised_ds = _subsample_labeled_dataset(
            syn_supervised_ds,
            n=cfg.stage3.combined_synth_train_size,
            num_classes=num_classes,
            seed=seed,
        )
        combined_ds = ConcatDataset([ref_supervised_ds, syn_supervised_ds])
        combined_loader = DataLoader(
            combined_ds,
            batch_size=int(cfg.loaders.synth_batch_size),
            shuffle=True,
            drop_last=bool(cfg.loaders.drop_last),
        )
        try:
            combined_stats = train_random_forest_classifier(
                combined_loader,
                test_loader=target_test_loader,
                seed=seed,
                name="Classifier/RF-ref+syn",
            )
        except RuntimeError:
            combined_clf = Classifier(d=d, num_classes=num_classes, hidden=list(cfg.stage3.hidden))
            combined_stats = train_classifier(
                combined_clf,
                combined_loader,
                test_loader=target_test_loader,
                epochs=int(cfg.stage3.epochs),
                lr=float(cfg.stage3.lr),
                device=device,
            )
        out["clf_loss_ref_plus_synth"] = float(combined_stats.get("clf_loss", nan))
        out["acc_ref_plus_synth"] = float(combined_stats.get("acc", nan))
        out["f1_ref_plus_synth"] = float(combined_stats.get("f1_macro", nan))
    return out
