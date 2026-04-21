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
    """
    Convert (epsilon, utility) points into a non-decreasing "best under budget" curve.

    For any privacy budget ε0, an algorithm that is (ε, δ)-DP for ε <= ε0 also
    satisfies (ε0, δ)-DP. Therefore the achievable utility as a function of ε
    can be represented as the prefix-max envelope over increasing ε.
    """
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
    parser = argparse.ArgumentParser(description="Overlay two privacy/utility curves from JSON sweep outputs.")
    parser.add_argument("--ours", required=True, help="JSON from NoisyFlow sweep (run.py outputs).")
    parser.add_argument("--baseline", required=True, help="JSON from IJCAI'19 DPOT sweep.")
    parser.add_argument("--label_ours", default="NoisyFlow", help="Legend label for our method.")
    parser.add_argument("--label_baseline", default="IJCAI'19 DPOT/DPDA", help="Legend label for baseline.")
    parser.add_argument("--output", default="tex/privacy_curve_compare_methods.pdf", help="Output PDF path.")
    parser.add_argument("--png", default=None, help="Optional PNG preview path.")
    parser.add_argument(
        "--monotone",
        action="store_true",
        help="Plot the monotone (prefix-max) envelope for each curve to remove stochastic dips.",
    )
    args = parser.parse_args()

    ours_metric, ours_points = _load_points(args.ours)
    base_metric, base_points = _load_points(args.baseline)
    if args.monotone:
        ours_points = _monotone_envelope(ours_points)
        base_points = _monotone_envelope(base_points)
    metric = ours_metric or base_metric
    if ours_metric and base_metric and ours_metric != base_metric:
        metric = "accuracy"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ox = [x for x, _ in ours_points]
    oy = [y for _, y in ours_points]
    bx = [x for x, _ in base_points]
    by = [y for _, y in base_points]

    ax.plot(ox, oy, marker="o", color="#4C78A8", label=args.label_ours)
    ax.plot(bx, by, marker="o", color="#F58518", label=args.label_baseline)
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
