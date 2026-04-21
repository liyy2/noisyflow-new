from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.config import load_config
from run import run_experiment, run_privacy_curve


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NoisyFlow YAML config and save stats as JSON.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda or cpu.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = str(args.device)

    if cfg.privacy_curve.enabled:
        payload: dict[str, Any] = {
            "config": args.config,
            "device": cfg.device,
            "kind": "privacy_curve",
            "results": run_privacy_curve(cfg, cfg.privacy_curve),
        }
    else:
        payload = {
            "config": args.config,
            "device": cfg.device,
            "kind": "experiment",
            "stats": run_experiment(cfg),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
