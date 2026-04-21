from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OKABE_ITO_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]


def _apply_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": True,
            "font.size": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.6,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.8,
            "legend.fontsize": 7,
            "legend.frameon": False,
        }
    )


def _build_stage_mia_features(cfg) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    from noisyflow.attacks.membership_inference import collect_stage_features
    from noisyflow.utils import set_seed
    from noisyflow.stage1.networks import VelocityField
    from noisyflow.stage1.training import sample_flow_euler, train_flow_stage1
    from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
    from noisyflow.stage2.training import train_ot_stage2, train_ot_stage2_cellot, train_ot_stage2_rectified_flow
    from run import _build_datasets, _infer_dims, _kernel_init_from_config, _split_dataset

    set_seed(cfg.seed)
    device = cfg.device

    client_datasets, target_ref, target_test = _build_datasets(cfg)
    d, num_classes = _infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)
    use_ot = cfg.stage2.option.upper() in {"A", "C"}

    target_loader = DataLoader(
        target_ref,
        batch_size=cfg.loaders.target_batch_size,
        shuffle=True,
        drop_last=cfg.loaders.drop_last,
    )

    member_feats: List[torch.Tensor] = []
    nonmember_feats: List[torch.Tensor] = []

    for idx, ds in enumerate(client_datasets):
        train_ds, holdout_ds = _split_dataset(
            ds,
            holdout_fraction=cfg.stage_mia.holdout_fraction,
            seed=cfg.stage_mia.seed + idx,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        flow = VelocityField(
            d=d,
            num_classes=num_classes,
            hidden=cfg.stage1.hidden,
            time_emb_dim=cfg.stage1.time_emb_dim,
            label_emb_dim=cfg.stage1.label_emb_dim,
        )
        train_flow_stage1(
            flow,
            train_loader,
            epochs=cfg.stage1.epochs,
            lr=cfg.stage1.lr,
            dp=cfg.stage1.dp,
            device=device,
        )

        def synth_sampler(batch_size: int, labels: torch.Tensor | None = None, flow_model=flow) -> torch.Tensor:
            if labels is None:
                labels = torch.randint(0, num_classes, (batch_size,), device=device)
            else:
                labels = labels.to(device).long().view(-1)
                if int(labels.numel()) != int(batch_size):
                    raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}")
            return sample_flow_euler(flow_model.to(device).eval(), labels, n_steps=cfg.stage2.flow_steps).cpu()

        real_x_loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        if cfg.stage2.cellot.enabled:
            if cfg.stage2.option.upper() != "A":
                raise ValueError("CellOT mode currently supports stage2.option A only.")
            kernel_init = _kernel_init_from_config(cfg.stage2.cellot.kernel_init)
            f = CellOTICNN(
                input_dim=d,
                hidden_units=cfg.stage2.cellot.hidden_units,
                activation=cfg.stage2.cellot.activation,
                softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
                softplus_beta=cfg.stage2.cellot.softplus_beta,
                fnorm_penalty=cfg.stage2.cellot.f_fnorm_penalty,
                kernel_init_fxn=kernel_init,
            )
            ot = CellOTICNN(
                input_dim=d,
                hidden_units=cfg.stage2.cellot.hidden_units,
                activation=cfg.stage2.cellot.activation,
                softplus_W_kernels=cfg.stage2.cellot.softplus_W_kernels,
                softplus_beta=cfg.stage2.cellot.softplus_beta,
                fnorm_penalty=cfg.stage2.cellot.g_fnorm_penalty,
                kernel_init_fxn=kernel_init,
            )
            train_ot_stage2_cellot(
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
        elif cfg.stage2.rectified_flow.enabled:
            ot = RectifiedFlowOT(
                d=d,
                hidden=cfg.stage2.rectified_flow.hidden,
                time_emb_dim=cfg.stage2.rectified_flow.time_emb_dim,
                act=cfg.stage2.rectified_flow.act,
                transport_steps=cfg.stage2.rectified_flow.transport_steps,
            )
            train_ot_stage2_rectified_flow(
                ot,
                source_loader=real_x_loader if cfg.stage2.option.upper() in {"A", "C"} else None,
                target_loader=target_loader,
                option=cfg.stage2.option,
                pair_by_label=cfg.stage2.pair_by_label,
                pair_by_ot=cfg.stage2.pair_by_ot,
                synth_sampler=synth_sampler if cfg.stage2.option.upper() in {"B", "C"} else None,
                epochs=cfg.stage2.epochs,
                lr=cfg.stage2.lr,
                dp=cfg.stage2.dp,
                device=device,
            )
        else:
            ot = ICNN(
                d=d,
                hidden=cfg.stage2.hidden,
                act=cfg.stage2.act,
                add_strong_convexity=cfg.stage2.add_strong_convexity,
            )
            train_ot_stage2(
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

        member_loader = DataLoader(
            train_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=False,
            drop_last=False,
        )
        nonmember_loader = DataLoader(
            holdout_ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=False,
            drop_last=False,
        )

        member_feats.append(
            collect_stage_features(
                flow,
                ot if use_ot else None,
                member_loader,
                use_ot=use_ot,
                num_flow_samples=cfg.stage_mia.num_flow_samples,
                include_ot_transport_norm=cfg.stage_mia.include_ot_transport_norm,
                seed=cfg.stage_mia.seed,
                device=device,
            )
        )
        nonmember_feats.append(
            collect_stage_features(
                flow,
                ot if use_ot else None,
                nonmember_loader,
                use_ot=use_ot,
                num_flow_samples=cfg.stage_mia.num_flow_samples,
                include_ot_transport_norm=cfg.stage_mia.include_ot_transport_norm,
                seed=cfg.stage_mia.seed + 123,
                device=device,
            )
        )

    all_member = torch.cat(member_feats, dim=0) if member_feats else torch.empty(0)
    all_nonmember = torch.cat(nonmember_feats, dim=0) if nonmember_feats else torch.empty(0)
    return all_member, all_nonmember, d, num_classes


def _run_stage_mia_attack(cfg) -> Dict[str, Any]:
    from noisyflow.attacks.membership_inference import run_stage_mia_attack
    from noisyflow.utils import set_seed

    set_seed(cfg.seed)
    member, nonmember, _, _ = _build_stage_mia_features(cfg)
    return run_stage_mia_attack(
        member,
        nonmember,
        attack_hidden=cfg.stage_mia.attack_hidden,
        attack_epochs=cfg.stage_mia.attack_epochs,
        attack_lr=cfg.stage_mia.attack_lr,
        attack_batch_size=cfg.stage_mia.attack_batch_size,
        attack_train_frac=cfg.stage_mia.attack_train_frac,
        max_samples=cfg.stage_mia.max_samples,
        seed=cfg.stage_mia.seed,
        return_curve=True,
        device=cfg.device,
    )


def _curve_from_stats(stats: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    auc = float(stats.get("stage_mia_attack_auc", float("nan")))
    fpr = np.asarray(stats.get("stage_mia_attack_fpr", []), dtype=np.float64)
    tpr = np.asarray(stats.get("stage_mia_attack_tpr", []), dtype=np.float64)
    if fpr.size == 0 or tpr.size == 0:
        raise RuntimeError("Missing ROC curve arrays. Ensure return_curve=True.")
    return fpr, tpr, auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage MIA ROC curves for two configs (DP vs non-DP).")
    parser.add_argument("--nodp-config", type=str, required=True, help="YAML config path for non-DP run.")
    parser.add_argument("--dp-config", type=str, required=True, help="YAML config path for DP run.")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/camelyon17_stage_mia_roc.pdf",
        help="Output PDF path (default: plots/camelyon17_stage_mia_roc.pdf).",
    )
    parser.add_argument("--json-output", type=str, default=None, help="Optional JSON path for ROC metrics.")
    args = parser.parse_args()

    from noisyflow.config import load_config

    nodp_cfg = load_config(args.nodp_config)
    dp_cfg = load_config(args.dp_config)
    if not nodp_cfg.stage_mia.enabled or not dp_cfg.stage_mia.enabled:
        raise ValueError("Both configs must set stage_mia.enabled: true")

    import matplotlib

    matplotlib.use("Agg")
    _apply_publication_style()

    color_nodp = OKABE_ITO_COLORS[4]  # blue
    color_dp = OKABE_ITO_COLORS[5]  # vermillion

    print(f"[ROC] Running non-DP config: {args.nodp_config}")
    nodp_stats = _run_stage_mia_attack(nodp_cfg)
    print(
        "[ROC] non-DP AUC={:.4f} acc={:.4f} advantage={:.4f}".format(
            float(nodp_stats.get("stage_mia_attack_auc", float("nan"))),
            float(nodp_stats.get("stage_mia_attack_acc", float("nan"))),
            float(nodp_stats.get("stage_mia_attack_advantage", float("nan"))),
        )
    )

    print(f"[ROC] Running DP config: {args.dp_config}")
    dp_stats = _run_stage_mia_attack(dp_cfg)
    print(
        "[ROC] DP     AUC={:.4f} acc={:.4f} advantage={:.4f}".format(
            float(dp_stats.get("stage_mia_attack_auc", float("nan"))),
            float(dp_stats.get("stage_mia_attack_acc", float("nan"))),
            float(dp_stats.get("stage_mia_attack_advantage", float("nan"))),
        )
    )

    fpr_nodp, tpr_nodp, auc_nodp = _curve_from_stats(nodp_stats)
    fpr_dp, tpr_dp, auc_dp = _curve_from_stats(dp_stats)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1.0, label="Chance")
    ax.plot(
        fpr_nodp,
        tpr_nodp,
        color=color_nodp,
        linestyle="-",
        linewidth=2.0,
        label=f"Non-DP (AUC={auc_nodp:.3f})",
    )
    ax.plot(
        fpr_dp,
        tpr_dp,
        color=color_dp,
        linestyle="--",
        linewidth=2.0,
        label=f"DP (AUC={auc_dp:.3f})",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out_path}")

    if args.json_output is not None:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodp_config": args.nodp_config,
            "dp_config": args.dp_config,
            "nodp": {
                "auc": auc_nodp,
                "acc": float(nodp_stats.get("stage_mia_attack_acc", float("nan"))),
                "advantage": float(nodp_stats.get("stage_mia_attack_advantage", float("nan"))),
            },
            "dp": {
                "auc": auc_dp,
                "acc": float(dp_stats.get("stage_mia_attack_acc", float("nan"))),
                "advantage": float(dp_stats.get("stage_mia_attack_advantage", float("nan"))),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
