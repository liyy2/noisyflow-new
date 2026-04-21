from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from statistics import mean, pstdev
from typing import Dict, Iterable, List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from noisyflow.config import load_config
from run import run_experiment


def _parse_csv(text: str) -> List[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(token) for token in _parse_csv(text)]


def _finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    xs = _finite(values)
    if not xs:
        return {"mean": float("nan"), "std": float("nan")}
    if len(xs) == 1:
        return {"mean": float(xs[0]), "std": 0.0}
    return {"mean": float(mean(xs)), "std": float(pstdev(xs))}


def _fmt(mu: float, sigma: float, digits: int = 4) -> str:
    if not (math.isfinite(mu) and math.isfinite(sigma)):
        return "nan"
    return f"{mu:.{digits}f} ± {sigma:.{digits}f}"


def _model_label(model_name: str) -> str:
    if model_name == "flow":
        return "DP-Flow + OT"
    if model_name == "vae":
        return "DP-VAE + OT"
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage I generator ablation (flow vs VAE) under a fixed Option B OT pipeline."
    )
    parser.add_argument("--config", required=True, help="Base experiment config. Stage II/III settings are reused as-is.")
    parser.add_argument("--models", default="flow,vae", help="Comma-separated Stage I models to compare.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds.")
    parser.add_argument("--output-json", default=None, help="Optional path to write raw runs + summary JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = [m.lower() for m in _parse_csv(args.models)]
    seeds = _parse_int_csv(args.seeds)

    if not seeds:
        raise ValueError("Provide at least one seed.")
    if not models:
        raise ValueError("Provide at least one stage1 model.")
    if str(cfg.stage2.option).upper() != "B":
        raise ValueError("This ablation should use stage2.option: B to isolate Stage I quality.")
    if cfg.stage2.dp is not None and cfg.stage2.dp.enabled:
        raise ValueError("Disable stage2.dp for this ablation; Option B should add no extra privacy cost.")
    raw_runs: List[Dict[str, float | int | str]] = []
    summary_rows: List[Dict[str, object]] = []
    for model_name in models:
        if model_name not in {"flow", "vae"}:
            raise ValueError(f"Unknown stage1 model '{model_name}'. Expected one of: flow, vae.")

        model_runs: List[Dict[str, float | int | str]] = []
        for seed in seeds:
            run_cfg = copy.deepcopy(cfg)
            run_cfg.seed = int(seed)
            if "seed" in run_cfg.data.params:
                run_cfg.data.params["seed"] = int(seed)
            run_cfg.stage1.model = model_name

            stats = run_experiment(run_cfg)
            row: Dict[str, float | int | str] = {
                "model": model_name,
                "seed": int(seed),
                "privacy": float(stats.get("epsilon_stage1_max", stats.get("epsilon_flow_max", float("nan")))),
                "sw2_x_tilde_nu": float(stats.get("sw2_synth_ref", float("nan"))),
                "sw2_y_tilde_nu": float(stats.get("sw2_synth_transported_ref", float("nan"))),
                "target_acc": float(stats.get("acc", float("nan"))),
                "acc_ref_plus_synth": float(stats.get("acc_ref_plus_synth", float("nan"))),
                "acc_ref_only": float(stats.get("acc_ref_only", float("nan"))),
                "acc_syn_raw": float(stats.get("acc_syn_raw", float("nan"))),
            }
            model_runs.append(row)
            raw_runs.append(row)

        privacy_stats = _mean_std(row["privacy"] for row in model_runs)
        sw2_raw_stats = _mean_std(row["sw2_x_tilde_nu"] for row in model_runs)
        sw2_transport_stats = _mean_std(row["sw2_y_tilde_nu"] for row in model_runs)
        acc_stats = _mean_std(row["target_acc"] for row in model_runs)
        acc_ref_plus_synth_stats = _mean_std(row["acc_ref_plus_synth"] for row in model_runs)
        acc_ref_only_stats = _mean_std(row["acc_ref_only"] for row in model_runs)
        acc_syn_raw_stats = _mean_std(row["acc_syn_raw"] for row in model_runs)

        summary_rows.append(
            {
                "stage1_model": _model_label(model_name),
                "privacy_mean": privacy_stats["mean"],
                "privacy_std": privacy_stats["std"],
                "sw2_x_tilde_nu_mean": sw2_raw_stats["mean"],
                "sw2_x_tilde_nu_std": sw2_raw_stats["std"],
                "sw2_y_tilde_nu_mean": sw2_transport_stats["mean"],
                "sw2_y_tilde_nu_std": sw2_transport_stats["std"],
                "target_acc_mean": acc_stats["mean"],
                "target_acc_std": acc_stats["std"],
                "acc_ref_plus_synth_mean": acc_ref_plus_synth_stats["mean"],
                "acc_ref_plus_synth_std": acc_ref_plus_synth_stats["std"],
                "acc_ref_only_mean": acc_ref_only_stats["mean"],
                "acc_ref_only_std": acc_ref_only_stats["std"],
                "acc_syn_raw_mean": acc_syn_raw_stats["mean"],
                "acc_syn_raw_std": acc_syn_raw_stats["std"],
                "num_seeds": len(model_runs),
            }
        )

    print("| Stage I model | Privacy | SW2(x~, nu) | SW2(y~, nu) | Target acc |")
    print("| --- | --- | --- | --- | --- |")
    for row in summary_rows:
        print(
            "| {stage1_model} | {privacy} | {sw2_raw} | {sw2_transport} | {acc} |".format(
                stage1_model=row["stage1_model"],
                privacy=_fmt(float(row["privacy_mean"]), float(row["privacy_std"])),
                sw2_raw=_fmt(float(row["sw2_x_tilde_nu_mean"]), float(row["sw2_x_tilde_nu_std"])),
                sw2_transport=_fmt(float(row["sw2_y_tilde_nu_mean"]), float(row["sw2_y_tilde_nu_std"])),
                acc=_fmt(float(row["target_acc_mean"]), float(row["target_acc_std"])),
            )
        )

    if args.output_json:
        payload = {
            "config": args.config,
            "models": models,
            "seeds": seeds,
            "runs": raw_runs,
            "summary": summary_rows,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved ablation summary to {args.output_json}")


if __name__ == "__main__":
    main()
