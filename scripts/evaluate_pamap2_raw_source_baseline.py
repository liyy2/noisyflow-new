from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.config import load_config
from noisyflow.stage3.training import train_random_forest_classifier
from run import _build_datasets, _infer_dims, _subsample_labeled_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PAMAP2 real source/no-transport RF baseline.")
    parser.add_argument("--config", default="configs/publication/pamap2_table_seed0.yaml")
    parser.add_argument("--out", default="results/publication_repro/pamap2_raw_real_source_baseline.json")
    parser.add_argument("--sizes", default="100,200,500,1000,2000,4000,all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client_datasets, target_ref, target_test = _build_datasets(cfg)
    _, num_classes = _infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)

    x_source = torch.cat([ds.tensors[0] for ds in client_datasets], dim=0)
    y_source = torch.cat([ds.tensors[1].long() for ds in client_datasets], dim=0)
    source_ds = TensorDataset(x_source, y_source)
    test_loader = DataLoader(target_test, batch_size=512, shuffle=False)

    rows = []
    for token in args.sizes.split(","):
        token = token.strip()
        n = None if token == "all" else int(token)
        train_ds = _subsample_labeled_dataset(source_ds, n=n, num_classes=num_classes, seed=cfg.seed)
        loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=False)
        stats = train_random_forest_classifier(
            loader,
            test_loader=test_loader,
            seed=cfg.seed,
            name=f"RF-source-real-{len(train_ds)}",
        )
        row = {"n": int(len(train_ds)), "acc": float(stats["acc"]), "f1_macro": float(stats["f1_macro"])}
        rows.append(row)
        print(row, flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
