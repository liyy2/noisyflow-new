from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from noisyflow.config import ExperimentConfig, load_config
from noisyflow.utils import DPConfig


def _ensure_dp(dp: Optional[DPConfig]) -> DPConfig:
    return dp if dp is not None else DPConfig(enabled=True)


def _set_target_eps(cfg: ExperimentConfig, *, stage: str, eps_total: float, stage2_fraction: float) -> None:
    stage = stage.strip().lower()
    if stage not in {"stage1", "stage2", "both"}:
        raise ValueError("stage must be one of: stage1, stage2, both")
    eps_total = float(eps_total)
    if eps_total <= 0.0:
        raise ValueError("eps_total must be > 0")

    stage2_fraction = float(stage2_fraction)
    if not (0.0 <= stage2_fraction <= 1.0):
        raise ValueError("stage2_fraction must be in [0, 1]")

    eps2 = eps_total * stage2_fraction
    eps1 = eps_total - eps2

    if stage in {"stage1", "both"}:
        cfg.stage1.dp = _ensure_dp(cfg.stage1.dp)
        cfg.stage1.dp.enabled = True
        cfg.stage1.dp.target_epsilon = float(eps1 if stage == "both" else eps_total)

    if stage in {"stage2", "both"}:
        cfg.stage2.dp = _ensure_dp(cfg.stage2.dp)
        cfg.stage2.dp.enabled = True
        cfg.stage2.dp.target_epsilon = float(eps2 if stage == "both" else eps_total)


def _sample_from(rng: random.Random, values: List[Any]) -> Any:
    if not values:
        raise ValueError("Empty search space")
    return values[int(rng.randrange(0, len(values)))]


def _apply_trial_hparams(cfg: ExperimentConfig, rng: random.Random) -> Dict[str, Any]:
    stage1_epochs = int(_sample_from(rng, [3, 5, 8, 12, 20, 30]))
    stage2_epochs = int(_sample_from(rng, [5, 10, 20, 30, 60, 120]))

    stage1_lr = float(_sample_from(rng, [3e-4, 5e-4, 1e-3, 2e-3]))
    stage2_lr = float(_sample_from(rng, [3e-4, 5e-4, 1e-3, 2e-3]))

    stage1_clip = float(_sample_from(rng, [0.1, 0.3, 1.0]))
    stage2_clip = float(_sample_from(rng, [0.1, 0.3, 1.0]))

    stage1_norm = str(_sample_from(rng, ["none", "layer"]))
    stage2_norm = str(_sample_from(rng, ["none", "layer"]))

    stage1_ema = _sample_from(rng, [None, 0.995, 0.999])
    stage2_ema = _sample_from(rng, [None, 0.995, 0.999])

    stage1_loss_norm = bool(_sample_from(rng, [False, True]))
    stage2_loss_norm = bool(_sample_from(rng, [False, True]))

    stage1_optim = str(_sample_from(rng, ["adam", "adamw"]))
    stage2_optim = str(_sample_from(rng, ["adam", "adamw"]))

    cfg.stage1.epochs = stage1_epochs
    cfg.stage1.lr = stage1_lr
    cfg.stage1.optimizer = stage1_optim
    cfg.stage1.mlp_norm = stage1_norm
    cfg.stage1.ema_decay = stage1_ema
    cfg.stage1.loss_normalize_by_dim = stage1_loss_norm
    if cfg.stage1.dp is not None:
        cfg.stage1.dp.max_grad_norm = stage1_clip

    cfg.stage2.epochs = stage2_epochs
    cfg.stage2.lr = stage2_lr
    cfg.stage2.optimizer = stage2_optim
    cfg.stage2.ema_decay = stage2_ema
    cfg.stage2.loss_normalize_by_dim = stage2_loss_norm
    cfg.stage2.rectified_flow.mlp_norm = stage2_norm
    if cfg.stage2.dp is not None:
        cfg.stage2.dp.max_grad_norm = stage2_clip

    return {
        "stage1": {
            "epochs": stage1_epochs,
            "lr": stage1_lr,
            "optimizer": stage1_optim,
            "mlp_norm": stage1_norm,
            "ema_decay": stage1_ema,
            "loss_normalize_by_dim": stage1_loss_norm,
            "max_grad_norm": stage1_clip,
        },
        "stage2": {
            "epochs": stage2_epochs,
            "lr": stage2_lr,
            "optimizer": stage2_optim,
            "mlp_norm": stage2_norm,
            "ema_decay": stage2_ema,
            "loss_normalize_by_dim": stage2_loss_norm,
            "max_grad_norm": stage2_clip,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-search tuner for low-epsilon NoisyFlow configs (uses dp.target_epsilon if Opacus supports it)."
    )
    parser.add_argument("--config", required=True, help="Base YAML config to start from.")
    parser.add_argument("--metric", default="acc_ref_plus_synth", help="Stats key to maximize.")
    parser.add_argument("--eps-total", type=float, required=True, help="Target total epsilon budget.")
    parser.add_argument(
        "--stage",
        default="both",
        choices=["stage1", "stage2", "both"],
        help="Which stage(s) should consume the epsilon budget.",
    )
    parser.add_argument(
        "--stage2-fraction",
        type=float,
        default=0.5,
        help="If --stage=both, allocate this fraction of eps_total to Stage2 (rest to Stage1).",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the tuner (not the experiment seed).")
    parser.add_argument("--out-json", default="tex/tune_low_epsilon_results.json", help="Write trial results JSON.")
    parser.add_argument(
        "--out-config",
        default="configs/auto_tuned_low_epsilon.yaml",
        help="Write best-found config YAML.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    metric = str(args.metric).strip()
    if not metric:
        raise ValueError("--metric must be a non-empty stats key (e.g., acc_ref_plus_synth)")
    if args.stage in {"stage2", "both"} and base_cfg.stage2.option.upper() not in {"A", "C"}:
        raise ValueError("--stage includes stage2 but base config stage2.option is not A or C")

    rng = random.Random(int(args.seed))

    results: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, ExperimentConfig]] = None

    # Import late so the script remains importable without torch/opacus installed.
    from run import run_experiment

    for t in range(1, int(args.trials) + 1):
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = int(base_cfg.seed)

        _set_target_eps(cfg, stage=args.stage, eps_total=float(args.eps_total), stage2_fraction=float(args.stage2_fraction))
        hparams = _apply_trial_hparams(cfg, rng=rng)

        stats = run_experiment(cfg)
        utility = stats.get(metric, None)
        eps = stats.get("epsilon_total_max", None)

        entry: Dict[str, Any] = {
            "trial": int(t),
            "hparams": hparams,
            "metric": metric,
            "utility": None if utility is None else float(utility),
            "epsilon": None if eps is None else float(eps),
            "epsilon_total_max": None if eps is None else float(eps),
            "epsilon_flow_max": None if stats.get("epsilon_flow_max") is None else float(stats["epsilon_flow_max"]),
            "epsilon_ot_max": None if stats.get("epsilon_ot_max") is None else float(stats["epsilon_ot_max"]),
        }
        results.append(entry)

        if utility is not None:
            utility_f = float(utility)
            if best is None or utility_f > best[0]:
                best = (utility_f, cfg)

        print(
            f"[Tune] {t:03d}/{int(args.trials)}  eps_total={entry['epsilon_total_max']}  {metric}={entry['utility']}"
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"stage": "tune_low_epsilon", "metric": metric, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"[Tune] Wrote results to {out_json}")

    if best is None:
        raise RuntimeError(f"No successful trials produced metric '{metric}'.")

    _best_utility, best_cfg = best
    out_cfg = Path(args.out_config)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to write the tuned config (pip install pyyaml).") from exc
    out_cfg.write_text(yaml.safe_dump(asdict(best_cfg), sort_keys=False), encoding="utf-8")
    print(f"[Tune] Best {metric}={_best_utility:.4f}  wrote config to {out_cfg}")


if __name__ == "__main__":
    main()
