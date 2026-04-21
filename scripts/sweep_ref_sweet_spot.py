from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from noisyflow.config import ExperimentConfig, load_config
from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import sample_flow_euler, train_flow_stage1
from noisyflow.stage2.networks import CellOTICNN, ICNN, RectifiedFlowOT
from noisyflow.stage2.training import train_ot_stage2, train_ot_stage2_cellot, train_ot_stage2_rectified_flow
from noisyflow.stage3.networks import Classifier
from noisyflow.stage3.training import server_synthesize_with_raw, train_classifier, train_random_forest_classifier
from noisyflow.utils import dp_label_prior_from_counts, set_seed, unwrap_model


def _parse_sizes(text: str) -> List[Optional[int]]:
    raw = [t.strip() for t in text.split(",") if t.strip()]
    out: List[Optional[int]] = []
    for token in raw:
        if token.lower() in {"all", "none"}:
            out.append(None)
        else:
            out.append(int(token))
    return out


def _train_ref_classifier(
    train_ds: TensorDataset,
    test_loader: DataLoader,
    *,
    batch_size: int,
    drop_last: bool,
    seed: int,
) -> float:
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    try:
        stats = train_random_forest_classifier(loader, test_loader=test_loader, seed=seed, name="RF")
        return float(stats.get("acc", float("nan")))
    except RuntimeError:
        d = int(train_ds.tensors[0].shape[1])
        num_classes = int(train_ds.tensors[1].max().item() + 1)
        clf = Classifier(d=d, num_classes=num_classes, hidden=[256, 256])
        stats = train_classifier(clf, loader, test_loader=test_loader, epochs=10, lr=1e-3, device="cpu")
        return float(stats.get("acc", float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ref-only vs ref+synth label budgets for a fixed trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--ref-sizes",
        type=str,
        default="10,25,50,75,100,200",
        help="Comma-separated ref_train_size values (e.g., '10,50,100').",
    )
    parser.add_argument(
        "--syn-sizes",
        type=str,
        default="500,1000,2000,4000,all",
        help="Comma-separated combined_synth_train_size values; use 'all' for full synthetic set.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write sweep results as JSON.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Optional path to write a PDF/PNG summary plot.",
    )
    parser.add_argument(
        "--plot-syn-size",
        type=str,
        default=None,
        help="Optional syn size (int or 'all') to plot ref+synth curve for, in addition to best-over-syn.",
    )
    parser.add_argument(
        "--min-ref",
        type=int,
        default=50,
        help="Minimum ref size when selecting a sweet spot (default: 50).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed (default: use config).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
        cfg.data.params["seed"] = int(args.seed)

    set_seed(cfg.seed)

    import run as run_module

    data_builder = run_module.data_builders.get(cfg.data.type)
    if data_builder is None:
        raise ValueError(f"Unknown data.type '{cfg.data.type}'")
    client_datasets, target_ref, target_test = data_builder(**cfg.data.params)
    d, num_classes = run_module._infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)

    target_loader = DataLoader(
        target_ref,
        batch_size=cfg.loaders.target_batch_size,
        shuffle=True,
        drop_last=cfg.loaders.drop_last,
    )
    target_test_loader = DataLoader(
        target_test,
        batch_size=cfg.loaders.test_batch_size,
        shuffle=False,
    )

    clients_out: List[Dict] = []
    for ds in client_datasets:
        train_ds = ds
        loader = DataLoader(
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
            loader,
            epochs=cfg.stage1.epochs,
            lr=cfg.stage1.lr,
            dp=cfg.stage1.dp,
            device=cfg.device,
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

        def synth_sampler(
            batch_size: int,
            labels: Optional[torch.Tensor] = None,
            flow=flow,
        ) -> torch.Tensor:
            if labels is None:
                labels = torch.randint(0, num_classes, (batch_size,), device=cfg.device)
            else:
                labels = labels.to(cfg.device).long().view(-1)
                if int(labels.numel()) != int(batch_size):
                    raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}")
            return sample_flow_euler(flow.to(cfg.device).eval(), labels, n_steps=cfg.stage2.flow_steps).cpu()

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
                raise ValueError("CellOT mode supports stage2.option A only.")
            kernel_init = run_module._kernel_init_from_config(cfg.stage2.cellot.kernel_init)
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
                device=cfg.device,
            )
        elif use_rectified_flow:
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
                device=cfg.device,
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
                device=cfg.device,
            )

        clients_out.append({"flow": unwrap_model(flow).cpu(), "ot": unwrap_model(ot).cpu(), "prior": prior})

    y_syn, l_syn, _x_syn_raw = server_synthesize_with_raw(
        clients_out,
        M_per_client=cfg.stage3.M_per_client,
        num_classes=num_classes,
        flow_steps=cfg.stage3.flow_steps,
        device=cfg.device,
    )

    if not (isinstance(target_ref, TensorDataset) and len(target_ref.tensors) >= 2):
        raise RuntimeError("target_ref must be a labeled TensorDataset for this sweep.")
    ref_full = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
    syn_full = TensorDataset(y_syn, l_syn.long())

    ref_sizes = _parse_sizes(args.ref_sizes)
    syn_sizes = _parse_sizes(args.syn_sizes)

    syn_only_acc_by_size: Dict[int, float] = {}
    for syn_n in syn_sizes:
        syn_ds = run_module._subsample_labeled_dataset(
            syn_full,
            n=syn_n,
            num_classes=num_classes,
            seed=cfg.seed,
        )
        syn_only_acc_by_size[int(len(syn_ds))] = _train_ref_classifier(
            syn_ds,
            target_test_loader,
            batch_size=cfg.loaders.synth_batch_size,
            drop_last=cfg.loaders.drop_last,
            seed=cfg.seed,
        )

    results: List[Dict[str, float]] = []
    for ref_n in ref_sizes:
        ref_ds = run_module._subsample_labeled_dataset(
            ref_full,
            n=ref_n,
            num_classes=num_classes,
            seed=cfg.seed,
        )
        acc_ref_only = _train_ref_classifier(
            ref_ds,
            target_test_loader,
            batch_size=cfg.loaders.target_batch_size,
            drop_last=cfg.loaders.drop_last,
            seed=cfg.seed,
        )
        for syn_n in syn_sizes:
            syn_ds = run_module._subsample_labeled_dataset(
                syn_full,
                n=syn_n,
                num_classes=num_classes,
                seed=cfg.seed,
            )
            acc_syn_only = float(syn_only_acc_by_size[int(len(syn_ds))])
            combined_ds = ConcatDataset([ref_ds, syn_ds])
            combined_loader = DataLoader(
                combined_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=cfg.loaders.drop_last,
            )
            try:
                combined_stats = train_random_forest_classifier(
                    combined_loader,
                    test_loader=target_test_loader,
                    seed=cfg.seed,
                    name="RF-ref+syn",
                )
                acc_ref_plus_synth = float(combined_stats.get("acc", float("nan")))
            except RuntimeError:
                combined_clf = Classifier(d=d, num_classes=num_classes, hidden=[256, 256])
                combined_stats = train_classifier(
                    combined_clf,
                    combined_loader,
                    test_loader=target_test_loader,
                    epochs=10,
                    lr=1e-3,
                    device="cpu",
                )
                acc_ref_plus_synth = float(combined_stats.get("acc", float("nan")))
            gain = acc_ref_plus_synth - acc_ref_only
            gain_vs_syn = acc_ref_plus_synth - acc_syn_only
            results.append(
                {
                    "ref_n": float(len(ref_ds)),
                    "syn_n": float(len(syn_ds)),
                    "acc_ref_only": float(acc_ref_only),
                    "acc_syn_only": float(acc_syn_only),
                    "acc_ref_plus_synth": float(acc_ref_plus_synth),
                    "gain": float(gain),
                    "gain_vs_syn": float(gain_vs_syn),
                }
            )

    results.sort(key=lambda r: (r["ref_n"], r["syn_n"]))
    print("\nref_n\tsyn_n\tacc_ref_only\tacc_syn_only\tacc_ref+syn\tgain\tgain_vs_syn")
    for r in results:
        print(
            f"{int(r['ref_n'])}\t{int(r['syn_n'])}\t{r['acc_ref_only']:.4f}\t{r['acc_syn_only']:.4f}\t{r['acc_ref_plus_synth']:.4f}\t{r['gain']:+.4f}\t{r['gain_vs_syn']:+.4f}"
        )

    candidates = [r for r in results if int(r["ref_n"]) >= int(args.min_ref)]
    if candidates:
        best = max(candidates, key=lambda r: (r["gain"], r["acc_ref_plus_synth"], r["gain_vs_syn"]))
        print(
            "\nSweet spot (ref_n>=min_ref): ref_n={} syn_n={} acc_ref_only={:.4f} acc_syn_only={:.4f} acc_ref+syn={:.4f} gain={:+.4f} gain_vs_syn={:+.4f}".format(
                int(best["ref_n"]),
                int(best["syn_n"]),
                best["acc_ref_only"],
                best["acc_syn_only"],
                best["acc_ref_plus_synth"],
                best["gain"],
                best["gain_vs_syn"],
            )
        )
    if syn_only_acc_by_size:
        best_syn_n, best_syn_acc = max(syn_only_acc_by_size.items(), key=lambda kv: kv[1])
        print(f"\nBest synth-only: syn_n={best_syn_n} acc_syn_only={best_syn_acc:.4f}")

    payload = {
        "config": str(args.config),
        "seed": int(cfg.seed),
        "ref_sizes": [None if r is None else int(r) for r in ref_sizes],
        "syn_sizes": [None if s is None else int(s) for s in syn_sizes],
        "syn_only": [{"syn_n": int(k), "acc_syn_only": float(v)} for k, v in sorted(syn_only_acc_by_size.items())],
        "results": results,
    }
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote sweep JSON to {args.output_json}")

    if args.plot_output is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

        ref_vals = sorted({int(r["ref_n"]) for r in results})
        syn_vals = sorted({int(r["syn_n"]) for r in results})

        ref_only_by_ref: Dict[int, float] = {}
        best_by_ref: Dict[int, Tuple[int, float, float]] = {}
        fixed_syn_by_ref: Dict[int, Tuple[int, float, float]] = {}

        fixed_syn_size: Optional[int] = None
        if args.plot_syn_size is not None:
            if str(args.plot_syn_size).strip().lower() == "all":
                fixed_syn_size = int(len(syn_full))
            else:
                fixed_syn_size = int(args.plot_syn_size)

        for ref_n in ref_vals:
            rows = [r for r in results if int(r["ref_n"]) == ref_n]
            if not rows:
                continue
            ref_only_by_ref[ref_n] = float(rows[0]["acc_ref_only"])
            best_row = max(rows, key=lambda r: float(r["gain"]))
            best_by_ref[ref_n] = (
                int(best_row["syn_n"]),
                float(best_row["acc_ref_plus_synth"]),
                float(best_row["gain"]),
            )
            if fixed_syn_size is not None:
                for r in rows:
                    if int(r["syn_n"]) == fixed_syn_size:
                        fixed_syn_by_ref[ref_n] = (
                            int(r["syn_n"]),
                            float(r["acc_ref_plus_synth"]),
                            float(r["gain"]),
                        )
                        break

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

        ax = axes[0]
        xs = np.array(ref_vals, dtype=float)
        ys_ref = np.array([ref_only_by_ref[r] for r in ref_vals], dtype=float)
        ys_best = np.array([best_by_ref[r][1] for r in ref_vals], dtype=float)

        ax.plot(xs, ys_ref, marker="o", linewidth=2, color="#4C78A8", label="Ref-only")
        ax.plot(
            xs,
            ys_best,
            marker="s",
            linewidth=2,
            color="#F58518",
            label="Ref+Synth (best gain)",
        )
        if syn_only_acc_by_size:
            best_syn_n, best_syn_acc = max(syn_only_acc_by_size.items(), key=lambda kv: kv[1])
            ax.axhline(
                best_syn_acc,
                linewidth=2,
                color="#B279A2",
                linestyle=":",
                label=f"Synth-only (best, syn={best_syn_n})",
            )
        if fixed_syn_size is not None and fixed_syn_by_ref:
            ys_fixed = np.array(
                [fixed_syn_by_ref.get(r, (fixed_syn_size, float("nan"), float("nan")))[1] for r in ref_vals],
                dtype=float,
            )
            ax.plot(
                xs,
                ys_fixed,
                marker="^",
                linewidth=2,
                linestyle="--",
                color="#54A24B",
                label=f"Ref+Synth (syn={fixed_syn_size})",
            )

        ax.set_xlabel("Labeled target budget $n_{\\mathrm{ref}}$")
        ax.set_ylabel("Target test accuracy")
        ax.set_title("Ref-only vs Ref+Synth")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="lower right", frameon=True)

        for r in ref_vals:
            syn_n, _acc, gain = best_by_ref[r]
            ax.text(
                float(r),
                float(best_by_ref[r][1]) + 0.01,
                f"{gain:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax = axes[1]
        mat = np.full((len(syn_vals), len(ref_vals)), np.nan, dtype=float)
        for i, syn_n in enumerate(syn_vals):
            for j, ref_n in enumerate(ref_vals):
                for row in results:
                    if int(row["syn_n"]) == int(syn_n) and int(row["ref_n"]) == int(ref_n):
                        mat[i, j] = float(row["gain"])
                        break
        vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(ref_vals)), [str(r) for r in ref_vals], rotation=0)
        ax.set_yticks(range(len(syn_vals)), [str(s) for s in syn_vals], rotation=0)
        ax.set_xlabel("$n_{\\mathrm{ref}}$")
        ax.set_ylabel("Synthetic labels used")
        ax.set_title("Gain heatmap ($\\Delta$ accuracy)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("$\\Delta$ = Acc(Ref+Synth) - Acc(Ref-only)")

        fig.tight_layout()
        fig.savefig(args.plot_output, dpi=200)
        print(f"Saved plot to {args.plot_output}")


if __name__ == "__main__":
    main()
