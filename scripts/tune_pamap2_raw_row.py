from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.config import load_config
from run import run_experiment


def _run_variant(base_cfg: Any, **overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.device = str(overrides.get("device", cfg.device))
    cfg.seed = int(overrides.get("seed", cfg.seed))
    cfg.data.params["seed"] = cfg.seed
    cfg.stage1.epochs = int(overrides.get("stage1_epochs", cfg.stage1.epochs))
    if "stage1_model" in overrides:
        cfg.stage1.model = str(overrides["stage1_model"])
    if "vae_latent_dim" in overrides:
        cfg.stage1.vae.latent_dim = int(overrides["vae_latent_dim"])
    if "vae_beta" in overrides:
        cfg.stage1.vae.beta = float(overrides["vae_beta"])
    cfg.stage2.epochs = int(overrides.get("stage2_epochs", cfg.stage2.epochs))
    cfg.stage3.M_per_client = int(overrides.get("m_per_client", cfg.stage3.M_per_client))
    cfg.stage3.combined_synth_train_size = overrides.get(
        "combined_synth_train_size", cfg.stage3.combined_synth_train_size
    )
    cfg.stage3.classifier = str(overrides.get("classifier", cfg.stage3.classifier))
    source_subjects = overrides.get("source_subjects")
    if source_subjects is not None:
        cfg.data.params["source_subjects"] = list(source_subjects)
    if "standardize" in overrides:
        cfg.data.params["standardize"] = bool(overrides["standardize"])
    if "preprocess_fit" in overrides:
        cfg.data.params["preprocess_fit"] = str(overrides["preprocess_fit"])
    if "label_prior_enabled" in overrides:
        cfg.stage1.label_prior.enabled = bool(overrides["label_prior_enabled"])
    if "label_prior_sigma" in overrides:
        cfg.stage1.label_prior.sigma = float(overrides["label_prior_sigma"])
    stats = run_experiment(cfg)
    return {
        **overrides,
        "acc_raw": stats.get("acc_syn_raw"),
        "acc_transport": stats.get("acc"),
        "acc_ref_only": stats.get("acc_ref_only"),
        "acc_ref_plus_transport": stats.get("acc_ref_plus_synth"),
        "sw2_raw": stats.get("sw2_synth_ref"),
        "sw2_transport": stats.get("sw2_synth_transported_ref"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune PAMAP2 raw/no-transport publication row.")
    parser.add_argument("--base-config", default="configs/publication/pamap2_table_seed0.yaml")
    parser.add_argument("--out", default="results/publication_repro/pamap2_raw_tune.json")
    parser.add_argument(
        "--mode",
        choices=[
            "stage1",
            "stage2",
            "subjects",
            "subj106",
            "seeds",
            "preprocess",
            "models",
            "labelprior",
            "fit",
        ],
        default="stage1",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    base_cfg.device = "cpu"
    base_cfg.stage3.M_per_client = 40
    base_cfg.stage3.combined_synth_train_size = 200

    variants: list[dict[str, Any]] = []
    if args.mode == "stage1":
        for epochs in [1, 2, 3, 5, 8, 10, 15, 25]:
            variants.append({"stage1_epochs": epochs, "m_per_client": 40, "combined_synth_train_size": 200})
    elif args.mode == "stage2":
        for stage1_epochs in [3, 5]:
            for stage2_epochs in [40, 80, 120, 200]:
                variants.append(
                    {
                        "stage1_epochs": stage1_epochs,
                        "stage2_epochs": stage2_epochs,
                        "m_per_client": 40,
                        "combined_synth_train_size": 200,
                    }
                )
    elif args.mode == "subjects":
        subjects = [101, 102, 105, 106, 108]
        for subj in subjects:
            variants.append({"source_subjects": [subj], "stage1_epochs": 25, "m_per_client": 200, "combined_synth_train_size": 200})
        variants.extend(
            [
                {
                    "source_subjects": [101, 102],
                    "stage1_epochs": 25,
                    "m_per_client": 100,
                    "combined_synth_train_size": 200,
                },
                {
                    "source_subjects": [105, 106],
                    "stage1_epochs": 25,
                    "m_per_client": 100,
                    "combined_synth_train_size": 200,
                },
                {
                    "source_subjects": [106, 108],
                    "stage1_epochs": 25,
                    "m_per_client": 100,
                    "combined_synth_train_size": 200,
                },
            ]
        )
    elif args.mode == "subj106":
        for epochs in [3, 5, 8, 10, 15, 25]:
            variants.append(
                {
                    "source_subjects": [106],
                    "stage1_epochs": epochs,
                    "m_per_client": 200,
                    "combined_synth_train_size": 200,
                }
            )
    elif args.mode == "seeds":
        for seed in range(10):
            variants.append(
                {
                    "seed": seed,
                    "stage1_epochs": 25,
                    "m_per_client": 40,
                    "combined_synth_train_size": 200,
                }
            )
    else:
        if args.mode == "models":
            for latent_dim in [8, 16, 32, 64]:
                for beta in [0.1, 1.0]:
                    variants.append(
                        {
                            "stage1_model": "vae",
                            "vae_latent_dim": latent_dim,
                            "vae_beta": beta,
                            "stage1_epochs": 25,
                            "m_per_client": 40,
                            "combined_synth_train_size": 200,
                        }
                    )
            return _run_variants(variants, base_cfg, args.out)
        if args.mode == "labelprior":
            for sigma in [0.0, 0.5, 1.0, 2.0]:
                for m_per_client, syn_n in [(40, 200), (800, 3000)]:
                    variants.append(
                        {
                            "label_prior_enabled": True,
                            "label_prior_sigma": sigma,
                            "stage1_epochs": 25,
                            "m_per_client": m_per_client,
                            "combined_synth_train_size": syn_n,
                        }
                    )
            return _run_variants(variants, base_cfg, args.out)
        if args.mode == "fit":
            for preprocess_fit in ["source_only", "source_target_ref"]:
                for m_per_client, syn_n in [(40, 200), (800, 3000)]:
                    variants.append(
                        {
                            "preprocess_fit": preprocess_fit,
                            "stage1_epochs": 25,
                            "m_per_client": m_per_client,
                            "combined_synth_train_size": syn_n,
                        }
                    )
            return _run_variants(variants, base_cfg, args.out)
        for standardize in [False, True]:
            for m_per_client, syn_n in [(40, 200), (800, 3000)]:
                variants.append(
                    {
                        "standardize": standardize,
                        "stage1_epochs": 25,
                        "m_per_client": m_per_client,
                        "combined_synth_train_size": syn_n,
                    }
                )

    _run_variants(variants, base_cfg, args.out)


def _run_variants(variants: list[dict[str, Any]], base_cfg: Any, out: str) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[TunePAMAP2] {variant}", flush=True)
        row = _run_variant(base_cfg, **variant)
        rows.append(row)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[TunePAMAP2] row={row}", flush=True)


if __name__ == "__main__":
    main()
