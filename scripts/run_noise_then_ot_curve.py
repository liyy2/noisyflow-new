from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _metric_label(metric: str) -> str:
    metric = metric.strip()
    if metric == "acc_ref_plus_synth":
        return "accuracy (ref+synth)"
    if metric == "acc_ref_only":
        return "accuracy (ref only)"
    if metric == "acc":
        return "accuracy (synth)"
    return metric


def _load_eps_from_json(path: str) -> List[float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Invalid 'results' list in {path}")
    eps: List[float] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        e = r.get("epsilon", None)
        if e is None:
            continue
        eps.append(float(e))
    eps = sorted({e for e in eps if e > 0.0})
    if not eps:
        raise ValueError(f"No epsilons found in {path}")
    return eps


def _parse_eps_list(values: Optional[Sequence[str]]) -> Optional[List[float]]:
    if values is None:
        return None
    eps: List[float] = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            eps.append(float(part))
    eps = sorted({e for e in eps if e > 0.0})
    return eps or None


def _compute_clip_norm_quantile(client_datasets, q: float) -> float:
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError("--clip-quantile must be in (0, 1]")
    norms: List[np.ndarray] = []
    for ds in client_datasets:
        x = ds.tensors[0].detach().cpu().numpy().astype(np.float64, copy=False)
        norms.append(np.linalg.norm(x, axis=1))
    all_norms = np.concatenate(norms, axis=0) if norms else np.asarray([], dtype=np.float64)
    if all_norms.size == 0:
        raise ValueError("Unable to compute clip_norm (no private samples)")
    return float(np.quantile(all_norms, q))


def _plot_privacy_curve(results: List[Dict[str, float]], output_path: str, metric: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    points = [(r["epsilon"], r["utility"]) for r in results if "epsilon" in r and "utility" in r]
    if not points:
        raise RuntimeError("No valid (epsilon, utility) points to plot.")
    points.sort(key=lambda x: float(x[0]))
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("epsilon (approx)")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved privacy-utility curve to {output_path}")


def _logspace(min_val: float, max_val: float, num: int) -> List[float]:
    if num <= 0:
        raise ValueError("num must be >= 1")
    min_val = float(min_val)
    max_val = float(max_val)
    if min_val <= 0.0 or max_val <= 0.0:
        raise ValueError("logspace requires min_val,max_val > 0")
    if num == 1:
        return [min_val]
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    return [10 ** (log_min + (log_max - log_min) * i / (num - 1)) for i in range(num)]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from noisyflow.baselines.noise_then_ot import NoiseThenOTConfig, run_noise_then_ot_experiment
    from noisyflow.config import load_config

    parser = argparse.ArgumentParser(
        description="Run 'noise private data then OT' baseline sweep on a NoisyFlow dataset."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config (uses data + stage2 + stage3 settings).")
    parser.add_argument("--metric", default="acc_ref_plus_synth", help="Stats key to plot (default: acc_ref_plus_synth).")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta for the Gaussian mechanism.")
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=None,
        help="Per-example L2 clip norm for data sanitization. If omitted, compute from --clip-quantile.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.95,
        help="Quantile of private sample norms used when --clip-norm is omitted.",
    )

    eps_group = parser.add_mutually_exclusive_group()
    eps_group.add_argument(
        "--eps",
        nargs="+",
        default=None,
        help="Explicit epsilons (space or comma separated), e.g. --eps 0.5 1 2 4 8.",
    )
    eps_group.add_argument(
        "--eps-from-json",
        default="plots/privacy_curve_noisyflow_stage2B_targeteps_full_envelope.json",
        help="JSON file whose 'results[].epsilon' values define the sweep grid (default: NoisyFlow Stage2=B envelope).",
    )
    eps_group.add_argument("--eps-min", type=float, default=0.5, help="Minimum epsilon for generated grid.")
    parser.add_argument("--eps-max", type=float, default=200.0, help="Maximum epsilon for generated grid.")
    parser.add_argument("--eps-num", type=int, default=12, help="Number of epsilons in generated grid.")
    parser.add_argument(
        "--eps-space",
        choices=["log", "linear"],
        default="log",
        help="Spacing for generated epsilon grid.",
    )

    parser.add_argument("--repeats", type=int, default=1, help="Runs per epsilon (seeds increment from base seed).")
    parser.add_argument("--out", default="plots/privacy_curve_noise_then_ot.json", help="Output JSON path.")
    parser.add_argument("--pdf", default="plots/privacy_curve_noise_then_ot.pdf", help="Output PDF path.")
    parser.add_argument("--png", default=None, help="Optional output PNG preview path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metric = str(args.metric).strip()
    if not metric:
        raise ValueError("--metric must be non-empty")

    eps_targets = _parse_eps_list(args.eps)
    if eps_targets is None:
        eps_from_json = getattr(args, "eps_from_json", None)
        if eps_from_json:
            eps_targets = _load_eps_from_json(str(eps_from_json))
        else:
            if args.eps_space == "linear":
                if int(args.eps_num) <= 0:
                    raise ValueError("--eps-num must be >= 1")
                if int(args.eps_num) == 1:
                    eps_targets = [float(args.eps_min)]
                else:
                    eps_targets = [
                        float(args.eps_min) + (float(args.eps_max) - float(args.eps_min)) * i / (int(args.eps_num) - 1)
                        for i in range(int(args.eps_num))
                    ]
            else:
                eps_targets = _logspace(float(args.eps_min), float(args.eps_max), int(args.eps_num))

    repeats = int(args.repeats)
    if repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    # Load data once (can be expensive).
    from run import _build_datasets

    client_datasets, target_ref, target_test = _build_datasets(cfg)
    if args.clip_norm is None:
        clip_norm = _compute_clip_norm_quantile(client_datasets, q=float(args.clip_quantile))
    else:
        clip_norm = float(args.clip_norm)
    if clip_norm <= 0.0:
        raise ValueError("--clip-norm must be > 0")

    results: List[Dict[str, float]] = []
    base_seed = int(cfg.seed)
    for eps in eps_targets:
        best: Optional[Dict[str, float]] = None
        for rep in range(repeats):
            cfg_rep = cfg
            cfg_rep.seed = base_seed + rep
            stats = run_noise_then_ot_experiment(
                client_datasets=client_datasets,
                target_ref=target_ref,
                target_test=target_test,
                cfg=cfg_rep,
                noise_cfg=NoiseThenOTConfig(
                    target_epsilon=float(eps),
                    delta=float(args.delta),
                    clip_norm=float(clip_norm),
                    seed=base_seed + rep,
                ),
            )
            utility = float(stats.get(metric, float("nan")))
            eps_realized = float(stats.get("epsilon_noise", float("nan")))
            entry = {
                "epsilon": eps_realized,
                "utility": utility,
                "target_epsilon": float(eps),
                "seed": float(cfg_rep.seed),
                "clip_norm": float(clip_norm),
                "noise_multiplier": float(stats.get("noise_multiplier_noise", float("nan"))),
            }
            if best is None or (not math.isnan(utility) and utility > float(best.get("utility", float("-inf")))):
                best = entry
            print(f"[NoiseThenOT] eps_target={eps:.3g} seed={cfg_rep.seed} eps={eps_realized:.3g} {metric}={utility:.4f}")
        if best is not None:
            results.append(best)

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "noise_then_ot",
        "metric": metric,
        "base_config": str(args.config),
        "delta": float(args.delta),
        "clip_norm": float(clip_norm),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved sweep JSON to {out_json}")

    out_pdf = Path(args.pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    _plot_privacy_curve(results, str(out_pdf), metric=metric)

    if args.png:
        out_png = Path(args.png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _plot_privacy_curve(results, str(out_png), metric=metric)


if __name__ == "__main__":
    main()
