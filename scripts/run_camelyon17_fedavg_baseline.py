from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.baselines.federated_classifier import train_fedavg_classifier
from noisyflow.baselines.dp_domain_adaptation import train_dp_erm_classifier_with_model
from noisyflow.config import load_config
from noisyflow.utils import DPConfig, set_seed
from run import _build_datasets, _infer_dims, _subsample_labeled_dataset


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (value != value):  # NaN check
        return "--"
    return f"{100.0 * float(value):.2f}"


def _format_privacy(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (value != value):
        return "--"
    return f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CAMELYON17-WILDS federated classifier baselines on the embedding pipeline."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a CAMELYON17 config YAML.")
    parser.add_argument("--ref-train-size", type=int, default=None, help="Override labeled target ref size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override local source training epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=None, help="Override target ref fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override classifier learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override classifier batch size.")
    parser.add_argument(
        "--dp",
        action="store_true",
        help="Train source-side local client models with DP-SGD before federated averaging.",
    )
    parser.add_argument("--noise-multiplier", type=float, default=2.0, help="DP noise multiplier.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="DP max grad norm.")
    parser.add_argument("--delta", type=float, default=1e-5, help="DP delta.")
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
    epochs = int(args.epochs or cfg.stage3.epochs)
    finetune_epochs = int(args.finetune_epochs or cfg.stage3.epochs)
    lr = float(args.lr or cfg.stage3.lr)
    ref_train_size = int(args.ref_train_size or cfg.stage3.ref_train_size or len(target_ref))
    hidden = list(cfg.stage3.hidden)

    ref_train = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
    ref_train = _subsample_labeled_dataset(ref_train, n=ref_train_size, num_classes=num_classes, seed=cfg.seed)

    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        for ds in client_datasets
    ]
    ref_loader = DataLoader(ref_train, batch_size=min(batch_size, len(ref_train)), shuffle=True, drop_last=False)
    test_loader = DataLoader(target_test, batch_size=max(batch_size, cfg.loaders.test_batch_size), shuffle=False)

    dp_cfg = None
    if args.dp:
        dp_cfg = DPConfig(
            enabled=True,
            max_grad_norm=float(args.max_grad_norm),
            noise_multiplier=float(args.noise_multiplier),
            delta=float(args.delta),
            grad_sample_mode="functorch",
            secure_mode=False,
        )

    results: Dict[str, object] = {
        "config": str(args.config),
        "device": device,
        "ref_train_size": ref_train_size,
        "local_epochs": epochs,
        "finetune_epochs": finetune_epochs,
        "lr": lr,
        "dp": bool(args.dp),
        "client_sizes": [int(len(ds)) for ds in client_datasets],
    }

    set_seed(cfg.seed)
    _, ref_only_stats = train_dp_erm_classifier_with_model(
        ref_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        epochs=finetune_epochs,
        lr=lr,
        dp=None,
        device=device,
        name="Baseline/Ref-only",
    )
    results["ref_only"] = ref_only_stats

    set_seed(cfg.seed)
    results["fedavg_source"] = train_fedavg_classifier(
        client_loaders,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        dp=dp_cfg,
        device=device,
        name="Baseline/FedAvg",
    )

    set_seed(cfg.seed)
    results["fedavg_source_then_ref_finetune"] = train_fedavg_classifier(
        client_loaders,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        dp=dp_cfg,
        ref_finetune_loader=ref_loader,
        ref_finetune_epochs=finetune_epochs,
        ref_finetune_lr=lr,
        device=device,
        name="Baseline/FedAvg",
    )

    print()
    print("CAMELYON17 federated classifier baselines")
    print("method\tacc(%)\tepsilon")
    print(f"ref_only\t{_format_metric(ref_only_stats.get('acc'))}\t--")
    fedavg_source = results["fedavg_source"]
    if isinstance(fedavg_source, dict):
        print(
            f"fedavg_source\t{_format_metric(fedavg_source.get('acc'))}\t"
            f"{_format_privacy(fedavg_source.get('epsilon_max'))}"
        )
    fedavg_ft = results["fedavg_source_then_ref_finetune"]
    if isinstance(fedavg_ft, dict):
        print(
            f"fedavg_source_then_ref_finetune\t{_format_metric(fedavg_ft.get('acc'))}\t"
            f"{_format_privacy(fedavg_ft.get('epsilon_max'))}"
        )

    print()
    print("Results:", json.dumps(results, indent=2))

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
