from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from noisyflow.config import ExperimentConfig, load_config


def _metric_label(metric: str) -> str:
    metric = metric.strip()
    if metric == "acc":
        return "accuracy (synth)"
    if metric == "acc_ref_only":
        return "accuracy (ref only)"
    if metric == "acc_ref_plus_synth":
        return "accuracy (ref+synth)"
    return metric


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


def _parse_eps_list(values: Sequence[str]) -> List[float]:
    eps: List[float] = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            eps.append(float(part))
    return eps


def _ensure_stage1_target_eps(cfg: ExperimentConfig, target_eps: float) -> None:
    if cfg.stage1.dp is None:
        raise ValueError("Base config must define stage1.dp for target-epsilon sweeps.")
    cfg.stage1.dp.enabled = True
    cfg.stage1.dp.target_epsilon = float(target_eps)


def _plot_points(
    points: List[Tuple[float, float]],
    *,
    metric: str,
    output_pdf: str,
    output_png: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    if title:
        ax.set_title(title)
    ax.set_xlabel("epsilon (approx)")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_pdf = Path(output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    print(f"[Sweep] Saved plot to {out_pdf}")

    if output_png:
        out_png = Path(output_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        print(f"[Sweep] Saved PNG preview to {out_png}")


def _monotone_envelope(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    points = sorted(points, key=lambda x: x[0])
    out: List[Tuple[float, float]] = []
    best = -float("inf")
    for eps, util in points:
        best = max(best, util)
        out.append((eps, best))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Stage1 dp.target_epsilon over a range (Stage2 post-processing recommended) and "
            "write raw + monotone-envelope privacy/utility curves."
        )
    )
    parser.add_argument("--config", required=True, help="Base YAML config to start from.")
    parser.add_argument("--metric", default="acc_ref_plus_synth", help="Stats key to maximize/plot.")
    parser.add_argument(
        "--resume-json",
        default=None,
        help="Optional existing raw sweep JSON (from this script) to resume from; missing eps targets are run and appended.",
    )

    eps_group = parser.add_mutually_exclusive_group()
    eps_group.add_argument(
        "--eps",
        nargs="+",
        default=None,
        help="Explicit target epsilons (space or comma separated), e.g. --eps 0.5 1 2 4 8.",
    )
    eps_group.add_argument("--eps-min", type=float, default=0.5, help="Minimum epsilon for generated grid.")
    parser.add_argument("--eps-max", type=float, default=20.0, help="Maximum epsilon for generated grid.")
    parser.add_argument("--eps-num", type=int, default=11, help="Number of epsilons in generated grid.")
    parser.add_argument(
        "--eps-space",
        choices=["log", "linear"],
        default="log",
        help="Spacing for generated epsilon grid.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per epsilon (seeds increment from base seed).",
    )
    parser.add_argument(
        "--ensure-monotone",
        action="store_true",
        help="If enabled, run extra seeds at dip points until best utility is non-decreasing (bounded by --max-extra-runs).",
    )
    parser.add_argument("--max-extra-runs", type=int, default=2, help="Extra runs for dip points when --ensure-monotone.")
    parser.add_argument("--tol", type=float, default=0.0, help="Allowed monotonicity slack (utility can dip by tol).")

    parser.add_argument("--out-json", default="plots/privacy_curve_stage2B_targeteps_full.json", help="Raw sweep JSON.")
    parser.add_argument(
        "--out-envelope-json",
        default="plots/privacy_curve_stage2B_targeteps_full_envelope.json",
        help="Monotone envelope JSON.",
    )
    parser.add_argument("--out-pdf", default="plots/privacy_curve_stage2B_targeteps_full.pdf", help="Raw plot PDF.")
    parser.add_argument(
        "--out-envelope-pdf",
        default="plots/privacy_curve_stage2B_targeteps_full_envelope.pdf",
        help="Envelope plot PDF.",
    )
    parser.add_argument("--png", default=None, help="Optional raw plot PNG path.")
    parser.add_argument("--envelope-png", default=None, help="Optional envelope plot PNG path.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    metric = str(args.metric).strip()
    if not metric:
        raise ValueError("--metric must be a non-empty stats key (e.g., acc_ref_plus_synth)")

    if args.eps is not None:
        eps_targets = _parse_eps_list(args.eps)
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

    eps_targets = sorted({float(e) for e in eps_targets if float(e) > 0.0})
    if not eps_targets:
        raise ValueError("No valid epsilons to run (must be > 0).")

    repeats = int(args.repeats)
    if repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    max_extra = int(args.max_extra_runs)
    if max_extra < 0:
        raise ValueError("--max-extra-runs must be >= 0")

    tol = float(args.tol)
    if tol < 0.0:
        raise ValueError("--tol must be >= 0")

    # Import late so the script remains importable without torch/opacus installed.
    from run import run_experiment

    sweep_runs: List[Dict[str, Any]] = []
    best_by_target: Dict[float, Dict[str, Any]] = {}

    if args.resume_json:
        resume_path = Path(str(args.resume_json))
        if resume_path.exists():
            payload = json.loads(resume_path.read_text(encoding="utf-8"))
            resume_metric = str(payload.get("metric", "")).strip()
            if resume_metric and resume_metric != metric:
                raise ValueError(f"--resume-json metric mismatch: {resume_metric} vs {metric}")
            resume_best = payload.get("best_points", [])
            if isinstance(resume_best, list):
                for bp in resume_best:
                    if not isinstance(bp, dict):
                        continue
                    tgt = bp.get("target_epsilon", None)
                    if tgt is None:
                        continue
                    try:
                        best_by_target[float(tgt)] = bp
                    except Exception:
                        continue
            resume_runs = payload.get("runs", [])
            if isinstance(resume_runs, list):
                sweep_runs.extend([r for r in resume_runs if isinstance(r, dict)])
            # Always keep existing points when resuming.
            eps_targets = sorted(set(eps_targets) | set(best_by_target.keys()))

    previous_best_utility: Optional[float] = None
    base_seed = int(base_cfg.seed)

    for target_eps in eps_targets:
        existing = best_by_target.get(float(target_eps))

        runs_for_eps: List[Dict[str, Any]] = []
        best_utility: Optional[float] = None
        best_stats: Optional[Dict[str, Any]] = None
        best_seed: Optional[int] = None
        best_eps: Optional[float] = None

        if isinstance(existing, dict):
            best_seed = existing.get("best_seed", None)
            best_utility = existing.get("utility", None)
            best_eps = existing.get("epsilon", None)
            existing_runs = existing.get("runs", [])
            if isinstance(existing_runs, list):
                runs_for_eps.extend([r for r in existing_runs if isinstance(r, dict)])

        total_runs = len(runs_for_eps)

        def _next_seed() -> int:
            used = []
            for r in runs_for_eps:
                s = r.get("seed", None)
                if s is None:
                    continue
                try:
                    used.append(int(s))
                except Exception:
                    continue
            if not used:
                return base_seed
            return max(used) + 1

        def _run_once(seed: int) -> None:
            nonlocal total_runs, best_utility, best_stats, best_seed, best_eps
            total_runs += 1

            cfg = copy.deepcopy(base_cfg)
            cfg.seed = int(seed)
            _ensure_stage1_target_eps(cfg, target_eps=float(target_eps))

            stats = run_experiment(cfg)
            utility = stats.get(metric, None)
            eps = stats.get("epsilon_total_max", None)

            run_entry = {
                "target_epsilon": float(target_eps),
                "seed": int(seed),
                "epsilon_total_max": None if eps is None else float(eps),
                "utility": None if utility is None else float(utility),
            }
            runs_for_eps.append(run_entry)
            sweep_runs.append(run_entry)

            if utility is not None:
                utility_f = float(utility)
                if best_utility is None or utility_f > float(best_utility):
                    best_utility = utility_f
                    best_stats = stats
                    best_seed = int(seed)
                    best_eps = None if eps is None else float(eps)

        # If we don't have a prior result for this target epsilon, run initial trials.
        if existing is None:
            for i in range(repeats):
                _run_once(seed=base_seed + i)

        # Optional extra runs to eliminate dips in the per-epsilon best curve.
        if args.ensure_monotone and previous_best_utility is not None:
            needed = float(previous_best_utility) - tol
            extra_i = 0
            while (best_utility is None or float(best_utility) < needed) and extra_i < max_extra:
                extra_i += 1
                _run_once(seed=_next_seed())

        # Summarize best point for this target eps.
        if best_utility is None:
            best_point = {
                "target_epsilon": float(target_eps),
                "best_seed": best_seed,
                "epsilon": best_eps,
                "utility": None,
                "runs": runs_for_eps,
            }
        else:
            eps_val = best_eps
            if eps_val is None and best_stats is not None:
                eps_val = best_stats.get("epsilon_total_max", None)
                eps_val = None if eps_val is None else float(eps_val)
            best_point = {
                "target_epsilon": float(target_eps),
                "best_seed": best_seed,
                "epsilon": eps_val,
                "utility": float(best_utility),
                "runs": runs_for_eps,
            }
            previous_best_utility = (
                float(best_utility)
                if previous_best_utility is None
                else max(float(previous_best_utility), float(best_utility))
            )

        best_by_target[float(target_eps)] = best_point

        print(f"[Sweep] target_eps={target_eps:.6g}  runs={total_runs}  best_{metric}={best_utility}")

    raw_points: List[Tuple[float, float]] = [
        (float(p["epsilon"]), float(p["utility"]))
        for p in best_by_target.values()
        if p.get("epsilon") is not None and p.get("utility") is not None
    ]
    if not raw_points:
        raise RuntimeError("All runs failed to produce valid epsilon+utility points.")
    raw_points.sort(key=lambda x: x[0])

    envelope_points = _monotone_envelope(raw_points)

    best_points: List[Dict[str, Any]] = [best_by_target[float(e)] for e in eps_targets if float(e) in best_by_target]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "stage": "sweep_target_epsilon",
                "metric": metric,
                "base_config": str(args.config),
                "best_points": best_points,
                "runs": sweep_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Sweep] Wrote raw sweep JSON to {out_json}")

    out_env_json = Path(args.out_envelope_json)
    out_env_json.parent.mkdir(parents=True, exist_ok=True)
    out_env_json.write_text(
        json.dumps(
            {
                "stage": "sweep_target_epsilon_envelope",
                "metric": metric,
                "note": "Prefix-max monotone envelope: best utility achievable under budget epsilon.",
                "base_config": str(args.config),
                "results": [{"epsilon": float(e), "utility": float(u)} for e, u in envelope_points],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Sweep] Wrote monotone envelope JSON to {out_env_json}")

    _plot_points(
        raw_points,
        metric=metric,
        output_pdf=str(args.out_pdf),
        output_png=args.png,
        title="NoisyFlow Stage2=B (raw best per ε target)",
    )
    _plot_points(
        envelope_points,
        metric=metric,
        output_pdf=str(args.out_envelope_pdf),
        output_png=args.envelope_png,
        title="NoisyFlow Stage2=B (monotone envelope)",
    )


if __name__ == "__main__":
    main()
