from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.baselines.dp_domain_adaptation import (
    DANNConfig,
    train_dp_dann,
    train_dp_dann_with_model,
    train_dp_erm_classifier,
    train_dp_erm_classifier_with_model,
)
from noisyflow.config import load_config
from noisyflow.utils import DPConfig, set_seed, unwrap_model
from run import _build_datasets, _infer_dims, _subsample_labeled_dataset


@torch.no_grad()
def _eval_logit_ensemble(models: List[torch.nn.Module], loader: DataLoader, *, device: str) -> float:
    if not models:
        return float("nan")
    for m in models:
        m.eval()

    n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).long()

        logits_sum: Optional[torch.Tensor] = None
        for m in models:
            inner = unwrap_model(m)
            if hasattr(inner, "predict"):
                logits = inner.predict(xb)  # type: ignore[no-any-return]
            else:
                out = m(xb)
                logits = out[0] if isinstance(out, tuple) else out
            logits_sum = logits if logits_sum is None else logits_sum + logits

        pred = logits_sum.argmax(dim=1) if logits_sum is not None else torch.zeros_like(yb)
        correct += int((pred == yb).sum().item())
        n += int(yb.numel())
    return float(correct / max(1, n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAMELYON17-WILDS DP domain-adaptation baselines.")
    parser.add_argument("--config", type=str, required=True, help="Path to a CAMELYON17 config YAML.")
    parser.add_argument("--ref-train-size", type=int, default=50, help="Labeled target ref size (default: 50).")
    parser.add_argument("--epochs", type=int, default=10, help="Baseline classifier epochs (default: 10).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Baseline classifier lr (default: 1e-3).")
    parser.add_argument("--batch-size", type=int, default=256, help="Baseline batch size (default: 256).")
    parser.add_argument("--noise-multiplier", type=float, default=2.0, help="DP noise multiplier (default: 2.0).")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="DP max grad norm (default: 1.0).")
    parser.add_argument("--delta", type=float, default=1e-5, help="DP delta (default: 1e-5).")
    parser.add_argument(
        "--dann-lambda",
        type=float,
        default=0.1,
        help="Gradient reversal strength for DP-DANN (default: 0.1).",
    )
    parser.add_argument(
        "--per-client",
        action="store_true",
        help="Train one DP model per client and aggregate via logit-ensemble; report epsilon_max across clients.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (default: from config).")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.data.type != "camelyon17_wilds":
        raise ValueError(f"Expected data.type=camelyon17_wilds, got {cfg.data.type}")

    device = str(args.device) if args.device is not None else cfg.device
    set_seed(cfg.seed)

    client_datasets, target_ref, target_test = _build_datasets(cfg)
    d, num_classes = _infer_dims(cfg, client_datasets)

    source_ds = ConcatDataset(client_datasets)
    test_loader = DataLoader(target_test, batch_size=1024, shuffle=False, drop_last=False)

    dp_cfg = DPConfig(
        enabled=True,
        max_grad_norm=float(args.max_grad_norm),
        noise_multiplier=float(args.noise_multiplier),
        delta=float(args.delta),
        grad_sample_mode="functorch",
        secure_mode=False,
    )

    results: Dict[str, Union[Dict[str, float], Dict[str, object]]] = {}

    if args.per_client:
        # DP-ERM per client, aggregate via ensemble.
        erm_models: List[torch.nn.Module] = []
        erm_client_stats: List[Dict[str, float]] = []
        erm_eps: List[float] = []
        for client_idx, ds in enumerate(client_datasets):
            set_seed(cfg.seed)
            client_loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
            model_i, stats_i = train_dp_erm_classifier_with_model(
                client_loader,
                test_loader,
                d=d,
                num_classes=num_classes,
                hidden=[256, 256],
                epochs=int(args.epochs),
                lr=float(args.lr),
                dp=dp_cfg,
                device=device,
                name=f"Baseline/DP-ERM-client{client_idx}",
            )
            erm_models.append(model_i)
            erm_client_stats.append(stats_i)
            if "epsilon" in stats_i:
                erm_eps.append(float(stats_i["epsilon"]))

        results["dp_erm_per_client"] = {
            "acc_ensemble": _eval_logit_ensemble(erm_models, test_loader, device=device),
            "epsilon_max": float(max(erm_eps)) if erm_eps else float("nan"),
            "epsilon_mean": float(sum(erm_eps) / max(1, len(erm_eps))) if erm_eps else float("nan"),
            "client_acc": [float(s.get("acc", float("nan"))) for s in erm_client_stats],
            "client_epsilon": [float(e) for e in erm_eps],
            "delta": float(dp_cfg.delta),
        }

        # DP-DANN per client, aggregate via ensemble.
        dann_models: List[torch.nn.Module] = []
        dann_client_stats: List[Dict[str, float]] = []
        dann_eps: List[float] = []
        target_x = target_ref.tensors[0].float()
        for client_idx, ds in enumerate(client_datasets):
            set_seed(cfg.seed)
            source_x = ds.tensors[0].float()
            source_y = ds.tensors[1].long()
            n_source = int(source_x.shape[0])
            n_target = int(target_x.shape[0])
            rep = max(1, (n_source + n_target - 1) // max(1, n_target))
            target_x_rep = target_x.repeat((rep, 1))[:n_source]

            x_train = torch.cat([source_x, target_x_rep], dim=0)
            y_class = torch.cat([source_y, -torch.ones((n_source,), dtype=torch.long)], dim=0)
            y_domain = torch.cat(
                [torch.zeros((n_source,), dtype=torch.long), torch.ones((n_source,), dtype=torch.long)],
                dim=0,
            )
            dann_train = TensorDataset(x_train, y_class, y_domain)
            model_i, stats_i = train_dp_dann_with_model(
                dann_train,
                test_loader,
                d=d,
                num_classes=num_classes,
                epochs=max(1, int(args.epochs)),
                lr=float(args.lr),
                dp=dp_cfg,
                batch_size=int(args.batch_size),
                device=device,
                cfg=DANNConfig(
                    feature_hidden=[256],
                    feature_dim=128,
                    label_hidden=[],
                    domain_hidden=[128],
                    lambda_domain=float(args.dann_lambda),
                ),
                name=f"Baseline/DP-DANN-client{client_idx}",
            )
            dann_models.append(model_i)
            dann_client_stats.append(stats_i)
            if "epsilon" in stats_i:
                dann_eps.append(float(stats_i["epsilon"]))

        results["dp_dann_per_client"] = {
            "acc_ensemble": _eval_logit_ensemble(dann_models, test_loader, device=device),
            "epsilon_max": float(max(dann_eps)) if dann_eps else float("nan"),
            "epsilon_mean": float(sum(dann_eps) / max(1, len(dann_eps))) if dann_eps else float("nan"),
            "client_acc": [float(s.get("acc", float("nan"))) for s in dann_client_stats],
            "client_epsilon": [float(e) for e in dann_eps],
            "delta": float(dp_cfg.delta),
        }

        print("Results:", json.dumps(results, indent=2))
        if args.out is not None:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Wrote {out_path}")
        return

    # DP-ERM on source only.
    set_seed(cfg.seed)
    source_loader = DataLoader(source_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    results["dp_erm_source"] = train_dp_erm_classifier(
        source_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=[256, 256],
        epochs=int(args.epochs),
        lr=float(args.lr),
        dp=dp_cfg,
        device=device,
        name="Baseline/DP-ERM-source",
    )

    # DP-ERM on source, then post-process fine-tune on public labeled ref.
    set_seed(cfg.seed)
    ref_supervised = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
    ref_supervised = _subsample_labeled_dataset(
        ref_supervised, n=int(args.ref_train_size), num_classes=num_classes, seed=cfg.seed
    )
    ref_loader = DataLoader(ref_supervised, batch_size=min(256, int(args.ref_train_size)), shuffle=True, drop_last=False)
    results["dp_erm_source_then_ref_finetune"] = train_dp_erm_classifier(
        source_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=[256, 256],
        epochs=int(args.epochs),
        lr=float(args.lr),
        dp=dp_cfg,
        ref_finetune_loader=ref_loader,
        ref_finetune_epochs=5,
        ref_finetune_lr=float(args.lr),
        device=device,
        name="Baseline/DP-ERM-source",
    )

    # DP-ERM on (source + small labeled ref). (Still DP-SGD for simplicity.)
    set_seed(cfg.seed)
    ref_supervised = TensorDataset(target_ref.tensors[0], target_ref.tensors[1].long())
    ref_supervised = _subsample_labeled_dataset(ref_supervised, n=int(args.ref_train_size), num_classes=num_classes, seed=cfg.seed)
    source_plus_ref = ConcatDataset([source_ds, ref_supervised])
    source_plus_ref_loader = DataLoader(source_plus_ref, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    results["dp_erm_source_plus_ref"] = train_dp_erm_classifier(
        source_plus_ref_loader,
        test_loader,
        d=d,
        num_classes=num_classes,
        hidden=[256, 256],
        epochs=int(args.epochs),
        lr=float(args.lr),
        dp=dp_cfg,
        device=device,
        name="Baseline/DP-ERM-source+ref",
    )

    # DP-DANN (unsupervised DA on public target_ref, balanced by repeating target).
    set_seed(cfg.seed)
    source_x = torch.cat([ds.tensors[0] for ds in client_datasets], dim=0).float()
    source_y = torch.cat([ds.tensors[1] for ds in client_datasets], dim=0).long()
    target_x = target_ref.tensors[0].float()

    n_source = int(source_x.shape[0])
    n_target = int(target_x.shape[0])
    rep = max(1, (n_source + n_target - 1) // max(1, n_target))
    target_x_rep = target_x.repeat((rep, 1))[:n_source]

    x_train = torch.cat([source_x, target_x_rep], dim=0)
    y_class = torch.cat([source_y, -torch.ones((n_source,), dtype=torch.long)], dim=0)
    y_domain = torch.cat([torch.zeros((n_source,), dtype=torch.long), torch.ones((n_source,), dtype=torch.long)], dim=0)
    dann_train = TensorDataset(x_train, y_class, y_domain)

    results["dp_dann"] = train_dp_dann(
        dann_train,
        test_loader,
        d=d,
        num_classes=num_classes,
        epochs=max(1, int(args.epochs)),
        lr=float(args.lr),
        dp=dp_cfg,
        batch_size=int(args.batch_size),
        device=device,
        cfg=DANNConfig(
            feature_hidden=[256],
            feature_dim=128,
            label_hidden=[],
            domain_hidden=[128],
            lambda_domain=float(args.dann_lambda),
        ),
        name="Baseline/DP-DANN",
    )

    print("Results:", json.dumps(results, indent=2))
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
