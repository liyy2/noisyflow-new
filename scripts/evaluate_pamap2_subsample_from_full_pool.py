from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.config import load_config
from scripts.rerun_paper_experiments_dp import _evaluate_for_sizes, _train_once


def main() -> None:
    cfg = load_config("configs/publication/pamap2_table_seed0.yaml")
    cfg.device = "cpu"
    cfg.stage1.epochs = 25
    cfg.stage2.epochs = 40
    cfg.stage3.M_per_client = 800
    cfg.stage3.combined_synth_train_size = 3000

    artifacts = _train_once(cfg)
    payload = _evaluate_for_sizes(
        artifacts,
        ref_sizes=[20],
        syn_sizes=[200, 500, 1000, 2000, 3000, 4000],
        include_raw=True,
    )
    out = Path("results/publication_repro/pamap2_eval_subsample_from_full_pool.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
