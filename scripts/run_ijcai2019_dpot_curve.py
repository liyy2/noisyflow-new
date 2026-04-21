from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _metric_label(metric: str) -> str:
    metric = metric.strip()
    if metric == "acc_ref_only":
        return "accuracy (ref only)"
    if metric == "acc_ref_plus_transport":
        return "accuracy (ref+transport)"
    return metric


def _plot_privacy_curve(results: List[Dict[str, Optional[float]]], output_path: str, metric: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    points = [(r["epsilon"], r["utility"]) for r in results if r.get("epsilon") is not None and r.get("utility") is not None]
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Run IJCAI'19 DPOT/DPDA baseline sweep on a NoisyFlow dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config (uses data + stage3 settings).")
    args = parser.parse_args()

    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required (pip install pyyaml).") from exc

    cfg_path = Path(args.config)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    dpot_raw = raw.get("ijcai2019_dpot", {}) or {}

    from noisyflow.baselines.ijcai2019_dpot import IJCai2019DPOTConfig, run_ijcai2019_dpot_experiment
    from noisyflow.config import load_config

    import run as run_module

    cfg = load_config(str(cfg_path))
    data_builder = run_module.data_builders.get(cfg.data.type)
    if data_builder is None:
        raise ValueError(f"Unsupported data.type for this baseline script: '{cfg.data.type}'")
    client_datasets, target_ref, target_test = data_builder(**cfg.data.params)
    d, num_classes = run_module._infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)

    device = str(dpot_raw.get("device", cfg.device))
    output_path = str(dpot_raw.get("output_path", "tex/ijcai2019_dpot_privacy_curve.pdf"))
    metric = str(dpot_raw.get("metric", "acc_ref_plus_transport"))
    noise_ratios = list(dpot_raw.get("noise_ratios", [0.1, 0.17, 0.3, 0.5]))

    dpot_cfg_base = IJCai2019DPOTConfig(
        projection_dim=int(dpot_raw.get("projection_dim", 30)),
        target_ot_size=int(dpot_raw["target_ot_size"]) if dpot_raw.get("target_ot_size", None) is not None else None,
        source_ot_size=int(dpot_raw["source_ot_size"]) if dpot_raw.get("source_ot_size", None) is not None else None,
        sinkhorn_reg=float(dpot_raw.get("sinkhorn_reg", 30.0)),
        sinkhorn_iters=int(dpot_raw.get("sinkhorn_iters", 200)),
        sinkhorn_eps=float(dpot_raw.get("sinkhorn_eps", 1e-9)),
        labelwise_ot=bool(dpot_raw.get("labelwise_ot", True)),
        noise_ratio=float(noise_ratios[0]) if noise_ratios else 0.3,
        delta=float(dpot_raw.get("delta", 1e-5)),
        label_epsilon=float(dpot_raw["label_epsilon"]) if dpot_raw.get("label_epsilon", None) is not None else None,
        source_clip_norm=float(dpot_raw["source_clip_norm"])
        if dpot_raw.get("source_clip_norm", None) is not None
        else None,
        classifier=str(dpot_raw.get("classifier", "auto")),
        device=device,
        seed=int(dpot_raw.get("seed", cfg.seed)),
    )

    results: List[Dict[str, Optional[float]]] = []
    for r in noise_ratios:
        dpot_cfg = IJCai2019DPOTConfig(**{**dpot_cfg_base.__dict__, "noise_ratio": float(r)})
        stats = run_ijcai2019_dpot_experiment(
            client_datasets=client_datasets,
            target_ref=target_ref,
            target_test=target_test,
            num_classes=num_classes,
            cfg=dpot_cfg,
            ref_train_size=int(cfg.stage3.ref_train_size or 50),
            combined_train_size=cfg.stage3.combined_synth_train_size,
            batch_size=cfg.loaders.synth_batch_size,
        )
        utility = float(stats.get(metric, float("nan")))
        eps = float(stats.get("epsilon_total", float("nan")))
        results.append({"noise_ratio": float(r), "epsilon": eps, "utility": utility})
        print(f"[IJCAI19-DPOT] r={r}  eps={eps:.3f}  {metric}={utility:.4f}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_privacy_curve(results, str(out_path), metric=metric)

    json_path = out_path.with_suffix(".json")
    payload = {"stage": "ijcai2019_dpot", "metric": metric, "results": results, "feature_dim": d, "num_classes": num_classes}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved sweep JSON to {json_path}")


if __name__ == "__main__":
    main()
