from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.baselines.dp_domain_adaptation import train_dp_erm_classifier_with_model
from noisyflow.baselines.fedgp import train_fedgp_classifier
from noisyflow.config import load_config
from noisyflow.utils import set_seed
from run import _build_datasets, _infer_dims, _subsample_labeled_dataset


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (value != value):
        return "--"
    return f"{100.0 * float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAMELYON17-WILDS predictor-only FedGP baseline on the embedding pipeline."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a CAMELYON17 config YAML.")
    parser.add_argument("--ref-train-size", type=int, default=None, help="Override labeled target ref size.")
    parser.add_argument("--rounds", type=int, default=25, help="Number of FedGP rounds.")
    parser.add_argument("--source-epochs", type=int, default=1, help="Local source epochs per round.")
    parser.add_argument("--target-epochs", type=int, default=1, help="Local target epochs per round.")
    parser.add_argument("--lr", type=float, default=None, help="Override classifier learning rate.")
    parser.add_argument("--server-lr", type=float, default=1.0, help="Server step size on aggregated updates.")
    parser.add_argument("--beta", type=float, default=0.5, help="FedGP beta weight.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override classifier batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override device (default: from config).")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.data.type != "camelyon17_wilds":
        raise ValueError(f"Expected data.type=camelyon17_wilds, got {cfg.data.type}")

    device = str(args.device) if args.device is not None else cfg.device
    set_seed(cfg.seed)

    client_datasets, target_ref, target_test = _build_datasets(cfg)
    d, num_classes = _infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)

    batch_size = int(args.batch_size or cfg.loaders.batch_size)
    lr = float(args.lr or cfg.stage3.lr)
    ref_train_size = int(args.ref_train_size or cfg.stage3.ref_train_size or len(target_ref))
    hidden = list(cfg.stage3.hidden)

    ref_train = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
    ref_train = _subsample_labeled_dataset(ref_train, n=ref_train_size, num_classes=num_classes, seed=cfg.seed)

    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False) for ds in client_datasets]
    ref_loader = DataLoader(ref_train, batch_size=min(batch_size, len(ref_train)), shuffle=True, drop_last=False)
    test_loader = DataLoader(target_test, batch_size=max(batch_size, cfg.loaders.test_batch_size), shuffle=False)

    results: Dict[str, object] = {
        "config": str(args.config),
        "device": device,
        "ref_train_size": ref_train_size,
        "rounds": int(args.rounds),
        "source_epochs": int(args.source_epochs),
        "target_epochs": int(args.target_epochs),
        "lr": lr,
        "server_lr": float(args.server_lr),
        "beta": float(args.beta),
        "client_sizes": [int(len(ds)) for ds in client_datasets],
    }

    set_seed(cfg.seed)
    _, ref_only_stats = train_dp_erm_classifier_with_model(
        ref_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        epochs=max(1, int(args.target_epochs) * int(args.rounds)),
        lr=lr,
        dp=None,
        device=device,
        name="Baseline/Ref-only",
    )
    results["ref_only"] = ref_only_stats

    set_seed(cfg.seed)
    results["fedgp"] = train_fedgp_classifier(
        client_loaders,
        ref_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        rounds=int(args.rounds),
        source_epochs=int(args.source_epochs),
        target_epochs=int(args.target_epochs),
        lr=lr,
        server_lr=float(args.server_lr),
        beta=float(args.beta),
        device=device,
        name="Baseline/FedGP",
    )

    print()
    print("CAMELYON17 FedGP baseline")
    print("method\tacc(%)")
    print(f"ref_only\t{_format_metric(ref_only_stats.get('acc'))}")
    fedgp_stats = results["fedgp"]
    if isinstance(fedgp_stats, dict):
        print(f"fedgp\t{_format_metric(fedgp_stats.get('acc'))}")

    print()
    print("Results:", json.dumps(results, indent=2))

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
