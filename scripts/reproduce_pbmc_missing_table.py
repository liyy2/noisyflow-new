from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noisyflow.config import load_config
from scripts.rerun_paper_experiments_dp import _evaluate_for_sizes, _train_once

TARGET_ROWS = {
    50: {"acc_ref_only": 0.770, "acc_transport_only": 0.792, "acc_ref_plus_transport": 0.798},
    10: {"acc_ref_only": 0.382, "acc_transport_only": 0.792, "acc_ref_plus_transport": 0.790},
    5: {"acc_ref_only": 0.258, "acc_transport_only": 0.792, "acc_ref_plus_transport": 0.793},
}
TARGET_EPSILON = 9.82


def _score_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    rows_by_ref = {int(row["ref_n"]): row for row in rows}
    ref_mae = mean(
        abs(float(rows_by_ref[ref_n]["acc_ref_only"]) - float(TARGET_ROWS[ref_n]["acc_ref_only"]))
        for ref_n in TARGET_ROWS
    )
    transport_mae = mean(
        abs(float(rows_by_ref[ref_n]["acc_transport_only"]) - float(TARGET_ROWS[ref_n]["acc_transport_only"]))
        for ref_n in TARGET_ROWS
    )
    combo_mae = mean(
        abs(float(rows_by_ref[ref_n]["acc_ref_plus_transport"]) - float(TARGET_ROWS[ref_n]["acc_ref_plus_transport"]))
        for ref_n in TARGET_ROWS
    )
    return {
        "ref_only_mae": float(ref_mae),
        "transport_only_mae": float(transport_mae),
        "ref_plus_transport_mae": float(combo_mae),
        "table_mae": float((ref_mae + transport_mae + combo_mae) / 3.0),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the missing PBMC few-shot table under fixed Option B and fixed synthetic budget."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/cellot_lupus_kang_missing_table_optionB_eps10_oldsplit.yaml",
            "configs/cellot_lupus_kang_missing_table_optionB_eps10_oldsplit_noprior.yaml",
            "configs/cellot_lupus_kang_missing_table_optionB_eps10_oldsplit_batch64_longot.yaml",
            "configs/cellot_lupus_kang_missing_table_optionB_eps10_oldsplit_batch64_moreot.yaml",
        ],
        help="Config paths to evaluate.",
    )
    parser.add_argument("--device", default="cuda", help="Device override.")
    parser.add_argument("--syn-size", type=int, default=500, help="Fixed transported-sample budget.")
    parser.add_argument("--ref-sizes", default="5,10,50", help="Comma-separated ref sizes.")
    parser.add_argument(
        "--out-dir",
        default="results/pbmc_missing_table_repro",
        help="Directory for per-config JSON outputs and the summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_sizes = [int(token.strip()) for token in str(args.ref_sizes).split(",") if token.strip()]
    if sorted(ref_sizes) != [5, 10, 50]:
        raise ValueError("This repro script expects ref sizes 5,10,50 to match the missing table.")

    summary: List[Dict[str, Any]] = []
    for config_path in args.configs:
        cfg = load_config(config_path)
        cfg.device = str(args.device)
        cfg.source_path = str(config_path)  # type: ignore[attr-defined]

        print(f"[Run] training {config_path} on device={cfg.device}")
        artifacts = _train_once(cfg)
        payload = _evaluate_for_sizes(
            artifacts,
            ref_sizes=ref_sizes,
            syn_sizes=[int(args.syn_size)],
            include_raw=False,
        )
        rows = sorted(payload["results"], key=lambda row: int(row["ref_n"]), reverse=True)
        scores = _score_rows(rows)
        eps = payload.get("epsilon_total_max")
        eps_gap = None if eps is None else abs(float(eps) - TARGET_EPSILON)

        record = {
            "config": str(config_path),
            "epsilon_total_max": eps,
            "epsilon_gap": eps_gap,
            "syn_size": int(args.syn_size),
            "rows": rows,
            "scores": scores,
        }
        summary.append(record)

        out_path = out_dir / f"{Path(config_path).stem}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            "[Done] {} eps={:.4f} ref_mae={:.4f} y_mae={:.4f} ref+y_mae={:.4f}".format(
                Path(config_path).name,
                float(eps) if eps is not None else float("nan"),
                scores["ref_only_mae"],
                scores["transport_only_mae"],
                scores["ref_plus_transport_mae"],
            )
        )
        torch.cuda.empty_cache()

    ranked = sorted(
        summary,
        key=lambda item: (
            float("inf") if item["scores"]["table_mae"] is None else item["scores"]["table_mae"],
            float("inf") if item["epsilon_gap"] is None else item["epsilon_gap"],
        ),
    )
    summary_payload = {"target_epsilon": TARGET_EPSILON, "target_rows": TARGET_ROWS, "ranked": ranked}
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\nTop matches")
    for item in ranked:
        print(
            "{}  eps={:.4f}  table_mae={:.4f}  y_mae={:.4f}  ref+y_mae={:.4f}".format(
                Path(item["config"]).name,
                float(item["epsilon_total_max"]) if item["epsilon_total_max"] is not None else float("nan"),
                float(item["scores"]["table_mae"]),
                float(item["scores"]["transport_only_mae"]),
                float(item["scores"]["ref_plus_transport_mae"]),
            )
        )


if __name__ == "__main__":
    main()
