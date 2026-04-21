from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run as run_module
from noisyflow.config import ExperimentConfig, load_config
from noisyflow.metrics import (
    label_silhouette_score,
    per_label_centroid_distances,
    same_label_domain_mixing,
)
from noisyflow.stage3.training import server_synthesize_with_raw
from noisyflow.utils import dp_label_prior_from_counts, set_seed, unwrap_model


def _parse_int_list(text: str) -> List[int]:
    return [int(t.strip()) for t in str(text).split(",") if t.strip()]


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def _clone_cfg_with_seed(cfg: ExperimentConfig, seed: int) -> ExperimentConfig:
    cfg_out = copy.deepcopy(cfg)
    cfg_out.seed = int(seed)
    cfg_out.data.params["seed"] = int(seed)
    return cfg_out


def _build_synth_sampler(
    cfg: ExperimentConfig,
    *,
    stage1_model: torch.nn.Module,
    stage1_model_name: str,
    num_classes: int,
    device: str,
):
    def synth_sampler(batch_size: int, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if labels is None:
            labels_local = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            labels_local = labels.to(device).long().view(-1)
            if int(labels_local.numel()) != int(batch_size):
                raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels_local.shape)}")
        if stage1_model_name == "flow":
            return run_module.sample_flow_euler(
                stage1_model.to(device).eval(),
                labels_local,
                n_steps=cfg.stage2.flow_steps,
            ).cpu()
        return run_module.sample_vae(stage1_model.to(device).eval(), labels_local).cpu()

    return synth_sampler


def _train_and_synthesize(cfg: ExperimentConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = cfg.device
    stage1_model_name = run_module._stage1_model_name(cfg)

    data_builder = run_module.data_builders.get(cfg.data.type)
    if data_builder is None:
        raise ValueError(f"Unknown data.type '{cfg.data.type}'")
    client_datasets, target_ref, target_test = data_builder(**cfg.data.params)
    d, num_classes = run_module._infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)

    if not (isinstance(target_test, TensorDataset) and len(target_test.tensors) >= 2):
        raise RuntimeError("target_test must be a labeled TensorDataset for structure-preservation analysis.")

    target_loader = DataLoader(
        target_ref,
        batch_size=cfg.loaders.target_batch_size,
        shuffle=True,
        drop_last=False,
    )

    clients_out: List[Dict[str, Any]] = []
    for ds in client_datasets:
        train_ds = ds
        loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        stage1_model = run_module._build_stage1_model(cfg, d=d, num_classes=num_classes)
        if stage1_model_name == "flow":
            run_module.train_flow_stage1(
                stage1_model,
                loader,
                epochs=cfg.stage1.epochs,
                lr=cfg.stage1.lr,
                optimizer=cfg.stage1.optimizer,
                weight_decay=cfg.stage1.weight_decay,
                ema_decay=cfg.stage1.ema_decay,
                loss_normalize_by_dim=cfg.stage1.loss_normalize_by_dim,
                dp=cfg.stage1.dp,
                device=device,
            )
        else:
            run_module.train_vae_stage1(
                stage1_model,
                loader,
                epochs=cfg.stage1.epochs,
                lr=cfg.stage1.lr,
                optimizer=cfg.stage1.optimizer,
                weight_decay=cfg.stage1.weight_decay,
                ema_decay=cfg.stage1.ema_decay,
                loss_normalize_by_dim=cfg.stage1.loss_normalize_by_dim,
                beta=cfg.stage1.vae.beta,
                dp=cfg.stage1.dp,
                device=device,
            )

        prior = None
        if cfg.stage1.label_prior.enabled:
            labels = train_ds.tensors[1]
            prior = dp_label_prior_from_counts(
                labels,
                num_classes=num_classes,
                mechanism=cfg.stage1.label_prior.mechanism,
                sigma=cfg.stage1.label_prior.sigma,
                device="cpu",
            )

        synth_sampler = _build_synth_sampler(
            cfg,
            stage1_model=stage1_model,
            stage1_model_name=stage1_model_name,
            num_classes=num_classes,
            device=device,
        )

        use_cellot = cfg.stage2.cellot.enabled
        use_rectified_flow = cfg.stage2.rectified_flow.enabled
        if use_cellot and use_rectified_flow:
            raise ValueError("Choose only one Stage2 model: stage2.cellot.enabled or stage2.rectified_flow.enabled.")

        real_x_loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        if use_cellot:
            if cfg.stage2.option.upper() != "A":
                raise ValueError("CellOT mode currently supports stage2.option A only.")
            kernel_init = run_module._kernel_init_from_config(cfg.stage2.cellot.kernel_init)
            f = run_module.CellOTICNN(
                input_dim=d,
                hidden_units=cfg.stage2.cellot.hidden_units,
                activation=cfg.stage2.cellot.activation,
                softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
                softplus_beta=cfg.stage2.cellot.softplus_beta,
                fnorm_penalty=cfg.stage2.cellot.f_fnorm_penalty,
                kernel_init_fxn=kernel_init,
            )
            ot = run_module.CellOTICNN(
                input_dim=d,
                hidden_units=cfg.stage2.cellot.hidden_units,
                activation=cfg.stage2.cellot.activation,
                softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
                softplus_beta=cfg.stage2.cellot.softplus_beta,
                fnorm_penalty=cfg.stage2.cellot.g_fnorm_penalty,
                kernel_init_fxn=kernel_init,
            )
            run_module.train_ot_stage2_cellot(
                f,
                ot,
                source_loader=real_x_loader,
                target_loader=target_loader,
                epochs=cfg.stage2.epochs,
                n_inner_iters=cfg.stage2.cellot.n_inner_iters,
                lr_f=cfg.stage2.lr,
                lr_g=cfg.stage2.lr,
                optim_cfg=cfg.stage2.cellot.optim,
                n_iters=cfg.stage2.cellot.n_iters,
                dp=cfg.stage2.dp,
                synth_sampler=synth_sampler,
                device=device,
            )
        elif use_rectified_flow:
            ot = run_module.RectifiedFlowOT(
                d=d,
                hidden=cfg.stage2.rectified_flow.hidden,
                time_emb_dim=cfg.stage2.rectified_flow.time_emb_dim,
                act=cfg.stage2.rectified_flow.act,
                transport_steps=cfg.stage2.rectified_flow.transport_steps,
                mlp_norm=cfg.stage2.rectified_flow.mlp_norm,
                mlp_dropout=cfg.stage2.rectified_flow.mlp_dropout,
            )
            run_module.train_ot_stage2_rectified_flow(
                ot,
                source_loader=real_x_loader if cfg.stage2.option.upper() in {"A", "C"} else None,
                target_loader=target_loader,
                option=cfg.stage2.option,
                pair_by_label=cfg.stage2.pair_by_label,
                pair_by_ot=cfg.stage2.pair_by_ot,
                pair_by_ot_method=cfg.stage2.pair_by_ot_method,
                synth_sampler=synth_sampler if cfg.stage2.option.upper() in {"B", "C"} else None,
                epochs=cfg.stage2.epochs,
                lr=cfg.stage2.lr,
                optimizer=cfg.stage2.optimizer,
                weight_decay=cfg.stage2.weight_decay,
                ema_decay=cfg.stage2.ema_decay,
                loss_normalize_by_dim=cfg.stage2.loss_normalize_by_dim,
                public_synth_steps=cfg.stage2.public_synth_steps,
                public_pretrain_epochs=cfg.stage2.public_pretrain_epochs,
                dp=cfg.stage2.dp,
                device=device,
            )
        else:
            ot = run_module.ICNN(
                d=d,
                hidden=cfg.stage2.hidden,
                act=cfg.stage2.act,
                add_strong_convexity=cfg.stage2.add_strong_convexity,
            )
            run_module.train_ot_stage2(
                ot,
                real_loader=real_x_loader if cfg.stage2.option.upper() in {"A", "C"} else None,
                target_loader=target_loader,
                option=cfg.stage2.option,
                synth_sampler=synth_sampler if cfg.stage2.option.upper() in {"B", "C"} else None,
                epochs=cfg.stage2.epochs,
                lr=cfg.stage2.lr,
                dp=cfg.stage2.dp,
                conj_steps=cfg.stage2.conj_steps,
                conj_lr=cfg.stage2.conj_lr,
                conj_clamp=cfg.stage2.conj_clamp,
                device=device,
            )

        clients_out.append(
            {
                "stage1_model": unwrap_model(stage1_model).cpu(),
                "stage1_model_type": stage1_model_name,
                "ot": unwrap_model(ot).cpu(),
                "prior": prior,
            }
        )

    y_syn, l_syn, x_syn_raw = server_synthesize_with_raw(
        clients_out,
        M_per_client=cfg.stage3.M_per_client,
        num_classes=num_classes,
        flow_steps=cfg.stage3.flow_steps,
        device=device,
    )

    return {
        "target_test_x": target_test.tensors[0].detach().cpu(),
        "target_test_y": target_test.tensors[1].long().detach().cpu(),
        "raw_x": x_syn_raw.detach().cpu(),
        "transported_x": y_syn.detach().cpu(),
        "synthetic_y": l_syn.long().detach().cpu(),
    }


def _balanced_label_indices(
    target_labels: torch.Tensor,
    synthetic_labels: torch.Tensor,
    *,
    per_label_cap: int = 150,
    min_per_label: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    target_np = target_labels.detach().cpu().numpy().reshape(-1)
    synthetic_np = synthetic_labels.detach().cpu().numpy().reshape(-1)
    rng = np.random.default_rng(int(seed))

    target_idx_all: List[int] = []
    synthetic_idx_all: List[int] = []
    kept_counts: Dict[int, int] = {}
    dropped_labels: List[int] = []

    shared = np.intersect1d(np.unique(target_np), np.unique(synthetic_np))
    for label in shared.tolist():
        target_idx = np.flatnonzero(target_np == label)
        synthetic_idx = np.flatnonzero(synthetic_np == label)
        n_keep = min(int(target_idx.size), int(synthetic_idx.size), int(per_label_cap))
        if n_keep < int(min_per_label):
            dropped_labels.append(int(label))
            continue
        rng.shuffle(target_idx)
        rng.shuffle(synthetic_idx)
        target_idx_all.extend(target_idx[:n_keep].tolist())
        synthetic_idx_all.extend(synthetic_idx[:n_keep].tolist())
        kept_counts[int(label)] = int(n_keep)

    target_idx_arr = np.array(target_idx_all, dtype=np.int64)
    synthetic_idx_arr = np.array(synthetic_idx_all, dtype=np.int64)
    if target_idx_arr.size > 0:
        rng.shuffle(target_idx_arr)
    if synthetic_idx_arr.size > 0:
        rng.shuffle(synthetic_idx_arr)

    return {
        "target_idx": target_idx_arr,
        "synthetic_idx": synthetic_idx_arr,
        "kept_counts": kept_counts,
        "dropped_labels": sorted(dropped_labels),
    }


def _evaluate_variant(
    *,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    variant_x: torch.Tensor,
    synthetic_y: torch.Tensor,
    n_neighbors: int,
) -> Dict[str, Any]:
    domains = torch.cat(
        [
            torch.zeros(target_x.shape[0], dtype=torch.long),
            torch.ones(variant_x.shape[0], dtype=torch.long),
        ],
        dim=0,
    )
    x_eval = torch.cat([target_x, variant_x], dim=0)
    y_eval = torch.cat([target_y, synthetic_y], dim=0)

    centroid_dists = per_label_centroid_distances(
        variant_x,
        synthetic_y,
        target_x,
        target_y,
    )
    centroid_mean = float(np.mean(list(centroid_dists.values()))) if centroid_dists else float("nan")

    return {
        "label_asw": float(label_silhouette_score(x_eval, y_eval)),
        f"same_label_domain_mixing_at_{int(n_neighbors)}": float(
            same_label_domain_mixing(x_eval, y_eval, domains, n_neighbors=n_neighbors)
        ),
        "centroid_target_distance_mean": centroid_mean,
        "centroid_target_distances": {str(k): float(v) for k, v in sorted(centroid_dists.items())},
    }


def _centroids_pca2(
    *,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    raw_x: torch.Tensor,
    transported_x: torch.Tensor,
    synthetic_y: torch.Tensor,
) -> Dict[str, Dict[str, List[float]]]:
    if target_x.dim() != 2 or target_x.shape[1] < 2:
        raise ValueError("Need at least two feature dimensions to plot PCA-2 centroids.")

    out: Dict[str, Dict[str, List[float]]] = {"target": {}, "raw": {}, "transported": {}}
    shared = sorted(
        set(target_y.detach().cpu().tolist()).intersection(set(synthetic_y.detach().cpu().tolist()))
    )
    for label in shared:
        t_mask = target_y == int(label)
        s_mask = synthetic_y == int(label)
        if int(t_mask.sum().item()) == 0 or int(s_mask.sum().item()) == 0:
            continue
        out["target"][str(int(label))] = target_x[t_mask, :2].mean(dim=0).detach().cpu().tolist()
        out["raw"][str(int(label))] = raw_x[s_mask, :2].mean(dim=0).detach().cpu().tolist()
        out["transported"][str(int(label))] = transported_x[s_mask, :2].mean(dim=0).detach().cpu().tolist()
    return out


def _plot_results(payload: Dict[str, Any], output_path: str, *, mixing_key: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    per_seed = payload["per_seed"]
    if not per_seed:
        raise RuntimeError("No per-seed results available to plot.")

    fig = plt.figure(figsize=(12.0, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    centroid_payload = payload["plot_seed"]["centroids_pca2"]
    labels = sorted(int(k) for k in centroid_payload["target"].keys())
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(labels):
        color = cmap(idx % 10)
        key = str(label)
        raw_xy = np.asarray(centroid_payload["raw"][key], dtype=np.float64)
        transport_xy = np.asarray(centroid_payload["transported"][key], dtype=np.float64)
        target_xy = np.asarray(centroid_payload["target"][key], dtype=np.float64)
        ax0.scatter(raw_xy[0], raw_xy[1], marker="o", s=70, color=color, alpha=0.9)
        ax0.scatter(transport_xy[0], transport_xy[1], marker="s", s=70, color=color, alpha=0.9)
        ax0.scatter(target_xy[0], target_xy[1], marker="X", s=85, color=color, alpha=0.95)
        ax0.annotate("", xy=transport_xy, xytext=raw_xy, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": color})
        ax0.annotate(
            "",
            xy=target_xy,
            xytext=transport_xy,
            arrowprops={"arrowstyle": "->", "lw": 1.2, "ls": "--", "color": color},
        )
        ax0.text(target_xy[0], target_xy[1], f" {label}", color=color, fontsize=9, va="center")

    ax0.set_title(f"Centroids in PCA-2 (seed={payload['plot_seed']['seed']})")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax0.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="", label="Raw synthetic"),
            Line2D([0], [0], marker="s", color="black", linestyle="", label="Transported synthetic"),
            Line2D([0], [0], marker="X", color="black", linestyle="", label="Target test"),
        ],
        loc="best",
        frameon=True,
    )

    def _metric_axis(ax, key: str, title: str, *, ylim: Optional[Tuple[float, float]] = None) -> None:
        raw_vals = np.asarray([float(row["raw"][key]) for row in per_seed], dtype=np.float64)
        transported_vals = np.asarray([float(row["transported"][key]) for row in per_seed], dtype=np.float64)
        means = [float(np.mean(raw_vals)), float(np.mean(transported_vals))]
        stds = [float(np.std(raw_vals, ddof=0)), float(np.std(transported_vals, ddof=0))]
        xs = np.array([0.0, 1.0], dtype=np.float64)
        ax.bar(xs, means, yerr=stds, width=0.6, color=["#7F7F7F", "#F58518"], alpha=0.85, capsize=4)
        jitter = np.linspace(-0.08, 0.08, raw_vals.size) if raw_vals.size > 1 else np.array([0.0])
        ax.scatter(np.full_like(raw_vals, xs[0]) + jitter, raw_vals, color="black", s=24, zorder=3)
        ax.scatter(np.full_like(transported_vals, xs[1]) + jitter, transported_vals, color="black", s=24, zorder=3)
        ax.set_xticks(xs, ["Raw", "Transported"])
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        if ylim is not None:
            ax.set_ylim(*ylim)

    _metric_axis(ax1, "label_asw", "Label ASW", ylim=(0.0, 1.0))
    _metric_axis(ax2, mixing_key, "Same-label domain mixing@15", ylim=(0.0, 1.0))

    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether transport preserves cell-type structure while improving target mixing."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated random seeds (default: 0,1,2).")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Optional path to write a PDF/PNG summary plot.",
    )
    parser.add_argument(
        "--per-label-cap",
        type=int,
        default=150,
        help="Maximum balanced samples per label and domain (default: 150).",
    )
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=10,
        help="Minimum balanced samples per label to keep that label (default: 10).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Neighbors for same-label domain mixing (default: 15).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = _parse_int_list(args.seeds)
    mixing_key = f"same_label_domain_mixing_at_{int(args.n_neighbors)}"

    per_seed: List[Dict[str, Any]] = []
    plot_seed: Optional[Dict[str, Any]] = None

    for seed in seeds:
        cfg_seed = _clone_cfg_with_seed(cfg, seed)
        bundle = _train_and_synthesize(cfg_seed)
        subset = _balanced_label_indices(
            bundle["target_test_y"],
            bundle["synthetic_y"],
            per_label_cap=args.per_label_cap,
            min_per_label=args.min_per_label,
            seed=seed,
        )
        target_idx = subset["target_idx"]
        synthetic_idx = subset["synthetic_idx"]
        if target_idx.size == 0 or synthetic_idx.size == 0:
            raise RuntimeError("Balanced evaluation subset is empty; lower min_per_label or check label overlap.")

        target_x_sub = bundle["target_test_x"].index_select(0, torch.from_numpy(target_idx))
        target_y_sub = bundle["target_test_y"].index_select(0, torch.from_numpy(target_idx))
        raw_x_sub = bundle["raw_x"].index_select(0, torch.from_numpy(synthetic_idx))
        transported_x_sub = bundle["transported_x"].index_select(0, torch.from_numpy(synthetic_idx))
        synthetic_y_sub = bundle["synthetic_y"].index_select(0, torch.from_numpy(synthetic_idx))

        raw_metrics = _evaluate_variant(
            target_x=target_x_sub,
            target_y=target_y_sub,
            variant_x=raw_x_sub,
            synthetic_y=synthetic_y_sub,
            n_neighbors=args.n_neighbors,
        )
        transported_metrics = _evaluate_variant(
            target_x=target_x_sub,
            target_y=target_y_sub,
            variant_x=transported_x_sub,
            synthetic_y=synthetic_y_sub,
            n_neighbors=args.n_neighbors,
        )

        row = {
            "seed": int(seed),
            "kept_label_counts": {str(k): int(v) for k, v in sorted(subset["kept_counts"].items())},
            "dropped_labels": [int(x) for x in subset["dropped_labels"]],
            "n_target_eval": int(target_x_sub.shape[0]),
            "n_synthetic_eval": int(synthetic_y_sub.shape[0]),
            "raw": raw_metrics,
            "transported": transported_metrics,
            "delta": {
                "label_asw": float(transported_metrics["label_asw"] - raw_metrics["label_asw"]),
                mixing_key: float(transported_metrics[mixing_key] - raw_metrics[mixing_key]),
                "centroid_target_distance_mean": float(
                    transported_metrics["centroid_target_distance_mean"] - raw_metrics["centroid_target_distance_mean"]
                ),
            },
        }
        per_seed.append(row)

        if plot_seed is None:
            plot_seed = {
                "seed": int(seed),
                "centroids_pca2": _centroids_pca2(
                    target_x=target_x_sub,
                    target_y=target_y_sub,
                    raw_x=raw_x_sub,
                    transported_x=transported_x_sub,
                    synthetic_y=synthetic_y_sub,
                ),
            }

        print(
            "[Structure] seed={}  label_asw(raw->{:.4f}, transport->{:.4f})  "
            "mix@{}(raw->{:.4f}, transport->{:.4f})  centroid(raw->{:.4f}, transport->{:.4f})".format(
                seed,
                raw_metrics["label_asw"],
                transported_metrics["label_asw"],
                int(args.n_neighbors),
                raw_metrics[mixing_key],
                transported_metrics[mixing_key],
                raw_metrics["centroid_target_distance_mean"],
                transported_metrics["centroid_target_distance_mean"],
            )
        )

    payload = {
        "config": str(args.config),
        "seeds": [int(s) for s in seeds],
        "per_label_cap": int(args.per_label_cap),
        "min_per_label": int(args.min_per_label),
        "n_neighbors": int(args.n_neighbors),
        "per_seed": per_seed,
        "aggregate": {
            "raw": {
                "label_asw": _stats(row["raw"]["label_asw"] for row in per_seed),
                mixing_key: _stats(row["raw"][mixing_key] for row in per_seed),
                "centroid_target_distance_mean": _stats(
                    row["raw"]["centroid_target_distance_mean"] for row in per_seed
                ),
            },
            "transported": {
                "label_asw": _stats(row["transported"]["label_asw"] for row in per_seed),
                mixing_key: _stats(row["transported"][mixing_key] for row in per_seed),
                "centroid_target_distance_mean": _stats(
                    row["transported"]["centroid_target_distance_mean"] for row in per_seed
                ),
            },
            "delta": {
                "label_asw": _stats(row["delta"]["label_asw"] for row in per_seed),
                mixing_key: _stats(row["delta"][mixing_key] for row in per_seed),
                "centroid_target_distance_mean": _stats(
                    row["delta"]["centroid_target_distance_mean"] for row in per_seed
                ),
            },
        },
        "plot_seed": plot_seed,
    }

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote JSON to {out_path}")

    if args.plot_output is not None:
        _plot_results(payload, args.plot_output, mixing_key=mixing_key)
        print(f"Wrote plot to {args.plot_output}")

    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
