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
    parser = argparse.ArgumentParser(description="Overlay four privacy/utility curves from JSON sweep outputs.")
    parser.add_argument("--noisyflow", required=True, help="JSON for NoisyFlow curve (e.g., Stage2=B envelope).")
    parser.add_argument("--noisyflow-tuned", required=True, help="JSON for tuned NoisyFlow curve.")
    parser.add_argument("--ijcai", required=True, help="JSON for IJCAI'19 baseline curve (e.g., unsupervised).")
    parser.add_argument("--ijcai-tuned", required=True, help="JSON for tuned IJCAI'19 curve.")

    parser.add_argument("--label_noisyflow", default="NoisyFlow", help="Legend label for NoisyFlow.")
    parser.add_argument("--label_noisyflow_tuned", default="NoisyFlow (tuned)", help="Legend label for tuned NoisyFlow.")
    parser.add_argument("--label_ijcai", default="IJCAI'19 DPOT", help="Legend label for IJCAI baseline.")
    parser.add_argument("--label_ijcai_tuned", default="IJCAI'19 DPOT (tuned)", help="Legend label for tuned IJCAI.")

    parser.add_argument("--output", default="plots/privacy_curve_compare_four_methods.pdf", help="Output PDF path.")
    parser.add_argument("--png", default=None, help="Optional PNG preview path.")
    parser.add_argument(
        "--monotone",
        action="store_true",
        help="Plot the monotone (prefix-max) envelope for each curve.",
    )
    args = parser.parse_args()

    nf_metric, nf_points = _load_points(args.noisyflow)
    nf_tuned_metric, nf_tuned_points = _load_points(args.noisyflow_tuned)
    ijcai_metric, ijcai_points = _load_points(args.ijcai)
    ijcai_tuned_metric, ijcai_tuned_points = _load_points(args.ijcai_tuned)

    if args.monotone:
        nf_points = _monotone_envelope(nf_points)
        nf_tuned_points = _monotone_envelope(nf_tuned_points)
        ijcai_points = _monotone_envelope(ijcai_points)
        ijcai_tuned_points = _monotone_envelope(ijcai_tuned_points)

    metrics = {m for m in [nf_metric, nf_tuned_metric, ijcai_metric, ijcai_tuned_metric] if m}
    metric = metrics.pop() if len(metrics) == 1 else "accuracy"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    series = [
        (args.label_noisyflow, nf_points, "#4C78A8"),
        (args.label_noisyflow_tuned, nf_tuned_points, "#72B7B2"),
        (args.label_ijcai, ijcai_points, "#F58518"),
        (args.label_ijcai_tuned, ijcai_tuned_points, "#E45756"),
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

