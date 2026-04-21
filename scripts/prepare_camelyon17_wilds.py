from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _parse_csv_list(value: str) -> List[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _select_indices_by_split_and_hospital(
    *,
    split_array: np.ndarray,
    hospital_array: np.ndarray,
    split_id: int,
    max_per_hospital: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    split_idx = np.flatnonzero(split_array == split_id)
    if max_per_hospital is None:
        return split_idx

    selected: List[np.ndarray] = []
    for hospital in np.unique(hospital_array[split_idx]).tolist():
        h = int(hospital)
        idx_h = split_idx[hospital_array[split_idx] == h]
        if idx_h.size <= max_per_hospital:
            selected.append(idx_h)
            continue
        chosen = rng.choice(idx_h, size=int(max_per_hospital), replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))
    if not selected:
        return np.empty((0,), dtype=np.int64)
    out = np.concatenate(selected, axis=0)
    out.sort()
    return out


def _build_resnet_feature_extractor(name: str) -> Tuple[torch.nn.Module, int, torch.nn.Module]:
    name = name.strip().lower()
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        return model, 512, weights.transforms()
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        return model, 2048, weights.transforms()
    raise ValueError(f"Unsupported model '{name}'. Choose from: resnet18, resnet50.")


def _iter_batches(loader: DataLoader, *, device: str) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("Expected WILDS batches as (x, y, metadata)")
        x, y, metadata = batch
        if not torch.is_tensor(x):
            raise ValueError("Expected x to be a tensor after transform; pass a torchvision transform to WILDSSubset.")
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True), metadata.to(device, non_blocking=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download CAMELYON17-WILDS (Koh et al., 2021) and precompute embedding features "
            "for NoisyFlow experiments."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="datasets/camelyon17_wilds/raw",
        help="Where to store the WILDS raw dataset files (default: datasets/camelyon17_wilds/raw).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/camelyon17_wilds/camelyon17_resnet18.npz",
        help="Output .npz path for features + metadata (default: datasets/camelyon17_wilds/camelyon17_resnet18.npz).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone used to embed patches (default: resnet18).",
    )
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="official",
        help="WILDS split scheme for CAMELYON17 (default: official).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,id_val,val,test",
        help="Comma-separated WILDS splits to include (default: train,id_val,val,test).",
    )
    parser.add_argument(
        "--max-per-hospital",
        type=int,
        default=20000,
        help="Optional cap per (split,hospital) group. Set <=0 to disable (default: 20000).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling (default: 0).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding extraction (default: 256).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for image loading (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedding extraction: cuda or cpu (default: cuda).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use AMP autocast on CUDA for faster embedding extraction.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")

    max_per_hospital: Optional[int] = int(args.max_per_hospital)
    if max_per_hospital is not None and max_per_hospital <= 0:
        max_per_hospital = None

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    model, feature_dim, transform = _build_resnet_feature_extractor(str(args.model))
    model.eval().to(device)

    from wilds import get_dataset
    from wilds.datasets.wilds_dataset import WILDSSubset

    dataset = get_dataset(
        dataset="camelyon17",
        download=True,
        root_dir=str(args.root_dir),
        split_scheme=str(args.split_scheme),
    )

    metadata_fields = list(dataset.metadata_fields)
    if "hospital" not in metadata_fields:
        raise RuntimeError(f"Expected 'hospital' in metadata_fields, got {metadata_fields}")
    hospital_col = int(metadata_fields.index("hospital"))

    split_array = np.asarray(dataset.split_array, dtype=np.int64)
    hospital_array = dataset.metadata_array[:, hospital_col].cpu().numpy().astype(np.int64, copy=False)

    splits = _parse_csv_list(str(args.splits))
    missing = [s for s in splits if s not in dataset.split_dict]
    if missing:
        raise ValueError(f"Unknown split(s) {missing}. Available: {sorted(dataset.split_dict.keys())}")

    rng = np.random.default_rng(int(args.seed))
    all_indices: List[np.ndarray] = []
    for split in splits:
        split_id = int(dataset.split_dict[split])
        idx = _select_indices_by_split_and_hospital(
            split_array=split_array,
            hospital_array=hospital_array,
            split_id=split_id,
            max_per_hospital=max_per_hospital,
            rng=rng,
        )
        print(f"[prep] split={split:6s} id={split_id} selected={int(idx.size)}")
        all_indices.append(idx)

    selected_idx = np.concatenate(all_indices, axis=0) if all_indices else np.empty((0,), dtype=np.int64)
    if selected_idx.size == 0:
        raise RuntimeError("No samples selected; check --splits/--max-per-hospital.")
    selected_idx.sort()

    subset = WILDSSubset(dataset, selected_idx, transform)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    n = int(selected_idx.size)
    x_out = np.empty((n, feature_dim), dtype=np.float32)
    y_out = np.empty((n,), dtype=np.int64)
    hospital_out = np.empty((n,), dtype=np.int64)
    split_out = split_array[selected_idx].astype(np.int64, copy=False)

    use_amp = bool(args.amp) and device.startswith("cuda")
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    offset = 0
    for xb, yb, mb in _iter_batches(loader, device=device):
        b = int(xb.shape[0])
        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            feats = model(xb)
        feats = feats.detach().float().cpu().numpy()
        if feats.ndim != 2 or feats.shape[1] != feature_dim:
            raise RuntimeError(f"Unexpected feature shape {feats.shape}, expected (B,{feature_dim})")

        x_out[offset : offset + b] = feats
        y_out[offset : offset + b] = yb.detach().long().cpu().numpy()
        hospital_out[offset : offset + b] = mb[:, hospital_col].detach().long().cpu().numpy()
        offset += b
        if offset % 4096 == 0 or offset == n:
            print(f"\r[prep] embedded {offset}/{n}", end="")
    print()

    payload: Dict[str, object] = {
        "dataset": "camelyon17",
        "paper": "Koh et al., 2021 (WILDS)",
        "model": str(args.model),
        "feature_dim": int(feature_dim),
        "root_dir": str(Path(args.root_dir).resolve()),
        "splits": splits,
        "split_dict": {k: int(v) for k, v in dataset.split_dict.items()},
        "metadata_fields": metadata_fields,
        "max_per_hospital": max_per_hospital,
        "seed": int(args.seed),
    }

    np.savez_compressed(
        output_path,
        X=x_out,
        label=y_out,
        hospital=hospital_out,
        split=split_out,
    )
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[prep] Saved: {output_path}  (N={n}, d={feature_dim})")
    print(f"[prep] Saved: {meta_path}")


if __name__ == "__main__":
    main()
