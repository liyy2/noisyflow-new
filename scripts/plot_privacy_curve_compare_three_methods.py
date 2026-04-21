from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def _load_points(path: str) -> Tuple[str, List[Tuple[float, float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metric = str(payload.get("metric", "")).strip()
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Invalid results list in {path}")
    points: List[Tuple[float, float]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        eps = r.get("epsilon", None)
        util = r.get("utility", None)
        if eps is None or util is None:
            continue
        points.append((float(eps), float(util)))
    if not points:
        raise ValueError(f"No valid points in {path}")
    points.sort(key=lambda x: x[0])
    return metric, points


def _monotone_envelope(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    best = float("-inf")
    out: List[Tuple[float, float]] = []
    for eps, util in sorted(points, key=lambda x: x[0]):
        best = max(best, float(util))
        out.append((float(eps), float(best)))
    return out


def _metric_label(metric: str) -> str:
    metric = metric.strip()
    if metric == "accuracy":
        return "accuracy"
    if metric == "acc_ref_plus_synth":
        return "accuracy (ref+synth)"
    if metric == "acc_ref_plus_transport":
        return "accuracy (ref+transport)"
    if metric == "acc_ref_only":
        return "accuracy (ref only)"
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay three privacy/utility curves from JSON sweep outputs.")
    parser.add_argument("--noisyflow", required=True, help="JSON for NoisyFlow curve.")
    parser.add_argument("--baseline", required=True, help="JSON for IJCAI'19 DPOT/DPDA baseline curve.")
    parser.add_argument("--noise-then-ot", required=True, help="JSON for noise-then-OT curve.")
    parser.add_argument("--label_noisyflow", default="NoisyFlow", help="Legend label for NoisyFlow.")
    parser.add_argument("--label_baseline", default="IJCAI'19 DPOT/DPDA", help="Legend label for baseline.")
    parser.add_argument("--label_noise_then_ot", default="Noise+OT (no generator)", help="Legend label for noise-then-OT.")
    parser.add_argument("--output", default="plots/privacy_curve_compare_three_methods.pdf", help="Output PDF path.")
    parser.add_argument("--png", default=None, help="Optional PNG preview path.")
    parser.add_argument(
        "--monotone",
        action="store_true",
        help="Plot the monotone (prefix-max) envelope for each curve.",
    )
    args = parser.parse_args()

    nf_metric, nf_points = _load_points(args.noisyflow)
    base_metric, base_points = _load_points(args.baseline)
    n2o_metric, n2o_points = _load_points(args.noise_then_ot)

    if args.monotone:
        nf_points = _monotone_envelope(nf_points)
        base_points = _monotone_envelope(base_points)
        n2o_points = _monotone_envelope(n2o_points)

    metric = nf_metric or base_metric or n2o_metric
    if nf_metric and base_metric and nf_metric != base_metric:
        metric = "accuracy"
    if n2o_metric and metric and n2o_metric != metric:
        metric = "accuracy"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    series = [
        (args.label_noisyflow, nf_points, "#4C78A8"),
        (args.label_baseline, base_points, "#F58518"),
        (args.label_noise_then_ot, n2o_points, "#54A24B"),
    ]
    for label, pts, color in series:
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, marker="o", color=color, label=label)
    ax.set_xlabel("epsilon (approx)")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")

    if args.png:
        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=200)
        print(f"Saved PNG preview to {png_path}")


if __name__ == "__main__":
    main()

