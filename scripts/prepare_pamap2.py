from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np


PAMAP2_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"

# PAMAP2 columns (0-indexed):
# 0: timestamp (s)
# 1: activity_id
# 2: heart_rate (bpm)
# Then 3 IMUs: hand (3..19), chest (20..36), ankle (37..53).
COL_ACTIVITY = 1
COL_HEART_RATE = 2

HAND_ACC16 = (4, 5, 6)
CHEST_ACC16 = (21, 22, 23)
ANKLE_ACC16 = (38, 39, 40)

HAND_GYRO = (10, 11, 12)
CHEST_GYRO = (27, 28, 29)
ANKLE_GYRO = (44, 45, 46)


def _parse_csv_ints(value: str) -> List[int]:
    items = [v.strip() for v in value.split(",")]
    out = []
    for item in items:
        if not item:
            continue
        out.append(int(item))
    return out


def _ffill_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={x.shape}")
    if x.size == 0:
        return x
    nan = ~np.isfinite(x)
    if nan.all():
        return np.zeros_like(x)
    first_valid = int(np.flatnonzero(~nan)[0])
    idx = np.where(nan, 0, np.arange(x.size, dtype=np.int64))
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[:first_valid] = x[first_valid]
    return out


def _majority_label(window_labels: np.ndarray) -> Tuple[int, float]:
    labels = np.asarray(window_labels).astype(np.int64, copy=False).reshape(-1)
    if labels.size == 0:
        return 0, 0.0
    counts = np.bincount(labels)
    major = int(counts.argmax())
    frac = float(counts[major]) / float(labels.size)
    return major, frac


def _window_stats_features(window: np.ndarray) -> np.ndarray:
    """
    window: (T, C)
    returns: (D,) where D = 6*C
    """
    w = np.asarray(window, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"Expected window shape (T,C), got {w.shape}")
    mean = w.mean(axis=0)
    std = w.std(axis=0)
    w_min = w.min(axis=0)
    w_max = w.max(axis=0)
    energy = (w * w).mean(axis=0)
    mad = np.abs(np.diff(w, axis=0)).mean(axis=0) if w.shape[0] > 1 else np.zeros_like(mean)
    return np.concatenate([mean, std, w_min, w_max, energy, mad], axis=0).astype(np.float32, copy=False)


def _download_if_missing(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"[pamap2] downloading: {url}")
    print(f"[pamap2] to: {dest}")
    urlretrieve(url, dest)


def _extract_zip_if_missing(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        return
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pamap2] extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)


@dataclass(frozen=True)
class Pamap2PrepConfig:
    protocol_dir: Path
    subjects: List[int]
    activities: List[int]
    include_gyro: bool
    include_heart_rate: bool
    downsample: int
    window_size: int
    stride: int
    label_purity: float
    representation: str
    max_windows_per_subject: Optional[int]


def _selected_columns(cfg: Pamap2PrepConfig) -> Tuple[List[int], List[str]]:
    cols: List[int] = [COL_ACTIVITY]
    names: List[str] = ["activity_id"]
    if cfg.include_heart_rate:
        cols.append(COL_HEART_RATE)
        names.append("heart_rate")

    cols.extend(list(HAND_ACC16))
    names.extend(["hand_acc16_x", "hand_acc16_y", "hand_acc16_z"])
    cols.extend(list(CHEST_ACC16))
    names.extend(["chest_acc16_x", "chest_acc16_y", "chest_acc16_z"])
    cols.extend(list(ANKLE_ACC16))
    names.extend(["ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z"])

    if cfg.include_gyro:
        cols.extend(list(HAND_GYRO))
        names.extend(["hand_gyro_x", "hand_gyro_y", "hand_gyro_z"])
        cols.extend(list(CHEST_GYRO))
        names.extend(["chest_gyro_x", "chest_gyro_y", "chest_gyro_z"])
        cols.extend(list(ANKLE_GYRO))
        names.extend(["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"])

    return cols, names


def _iter_subject_windows(
    subject_path: Path,
    cfg: Pamap2PrepConfig,
    activity_to_label: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    usecols, col_names = _selected_columns(cfg)
    data = np.loadtxt(subject_path, usecols=usecols, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array from {subject_path}, got {data.shape}")
    if data.shape[1] != len(usecols):
        raise ValueError(f"Expected {len(usecols)} columns from {subject_path}, got {data.shape[1]}")

    data = data[:: int(cfg.downsample)]
    activity = data[:, 0].astype(np.int64, copy=False)
    feats = data[:, 1:]

    if cfg.include_heart_rate:
        hr_idx = int(col_names.index("heart_rate") - 1)  # exclude activity_id
        feats[:, hr_idx] = _ffill_nan_1d(feats[:, hr_idx])

    finite = np.isfinite(feats).all(axis=1)
    feats = feats[finite]
    activity = activity[finite]

    if int(feats.shape[0]) < int(cfg.window_size):
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    windows: List[np.ndarray] = []
    labels: List[int] = []
    for start in range(0, int(feats.shape[0]) - int(cfg.window_size) + 1, int(cfg.stride)):
        a = activity[start : start + int(cfg.window_size)]
        major, frac = _majority_label(a)
        if frac < float(cfg.label_purity):
            continue
        if major not in activity_to_label:
            continue
        w = feats[start : start + int(cfg.window_size)]
        if cfg.representation == "flat":
            x = w.reshape(-1).astype(np.float32, copy=False)
        elif cfg.representation == "stats":
            x = _window_stats_features(w)
        else:
            raise ValueError(f"Unknown representation: {cfg.representation}")
        windows.append(x)
        labels.append(int(activity_to_label[major]))
        if cfg.max_windows_per_subject is not None and len(labels) >= int(cfg.max_windows_per_subject):
            break

    if not windows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack(windows, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess PAMAP2 (Protocol) into NoisyFlow-ready window features."
    )
    parser.add_argument(
        "--raw-zip",
        type=str,
        default="datasets/pamap2/raw/PAMAP2_Dataset.zip",
        help="Path for the downloaded PAMAP2 zip (default: datasets/pamap2/raw/PAMAP2_Dataset.zip).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="datasets/pamap2/raw/PAMAP2_Dataset",
        help="Directory containing extracted PAMAP2_Dataset/ (default: datasets/pamap2/raw/PAMAP2_Dataset).",
    )
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default=None,
        help="Directory containing subject*.dat Protocol files (default: <raw-dir>/Protocol).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="101,102,103,104,105,106,107,108,109",
        help="Comma-separated subject ids (default: 101..109).",
    )
    parser.add_argument(
        "--activities",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated activity ids to keep (default: 1,2,3,4,5,6).",
    )
    parser.add_argument("--include-gyro", action="store_true", help="Include gyroscope channels.")
    parser.add_argument("--no-heart-rate", action="store_true", help="Drop heart rate channel.")
    parser.add_argument("--downsample", type=int, default=4, help="Keep every Nth row (default: 4).")
    parser.add_argument("--window-size", type=int, default=128, help="Window length in samples (default: 128).")
    parser.add_argument("--stride", type=int, default=64, help="Stride in samples (default: 64).")
    parser.add_argument(
        "--label-purity",
        type=float,
        default=0.9,
        help="Require majority label fraction in each window (default: 0.9).",
    )
    parser.add_argument(
        "--representation",
        choices=["stats", "flat"],
        default="stats",
        help="Window representation: stats (tabular) or flat (raw flatten) (default: stats).",
    )
    parser.add_argument(
        "--max-windows-per-subject",
        type=int,
        default=None,
        help="Optional cap to limit windows per subject (default: no cap).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/pamap2/pamap2_protocol_windows.npz",
        help="Output .npz path (default: datasets/pamap2/pamap2_protocol_windows.npz).",
    )
    args = parser.parse_args()

    raw_zip = Path(args.raw_zip)
    raw_dir = Path(args.raw_dir)
    protocol_dir = Path(args.protocol_dir) if args.protocol_dir else raw_dir / "Protocol"

    _download_if_missing(PAMAP2_ZIP_URL, raw_zip)
    _extract_zip_if_missing(raw_zip, raw_dir)
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    subjects = _parse_csv_ints(str(args.subjects))
    activities = _parse_csv_ints(str(args.activities))
    if not activities:
        raise ValueError("activities must be non-empty")
    activity_to_label = {int(a): i for i, a in enumerate(sorted(set(activities)))}

    cfg = Pamap2PrepConfig(
        protocol_dir=protocol_dir,
        subjects=subjects,
        activities=sorted(set(activities)),
        include_gyro=bool(args.include_gyro),
        include_heart_rate=not bool(args.no_heart_rate),
        downsample=int(args.downsample),
        window_size=int(args.window_size),
        stride=int(args.stride),
        label_purity=float(args.label_purity),
        representation=str(args.representation),
        max_windows_per_subject=int(args.max_windows_per_subject)
        if args.max_windows_per_subject is not None
        else None,
    )
    if cfg.downsample <= 0:
        raise ValueError("downsample must be > 0")
    if cfg.window_size <= 1:
        raise ValueError("window_size must be > 1")
    if cfg.stride <= 0:
        raise ValueError("stride must be > 0")
    if not (0.0 < cfg.label_purity <= 1.0):
        raise ValueError("label_purity must be in (0,1]")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subjects: List[np.ndarray] = []
    for subj in cfg.subjects:
        subject_path = cfg.protocol_dir / f"subject{subj}.dat"
        if not subject_path.exists():
            print(f"[pamap2] WARNING: missing {subject_path}; skipping")
            continue
        X_s, y_s = _iter_subject_windows(subject_path, cfg, activity_to_label)
        if y_s.size == 0:
            print(f"[pamap2] subject={subj} windows=0 (after filtering); skipping")
            continue
        all_X.append(X_s)
        all_y.append(y_s)
        all_subjects.append(np.full((int(y_s.shape[0]),), int(subj), dtype=np.int64))
        print(f"[pamap2] subject={subj} windows={int(y_s.shape[0])} d={int(X_s.shape[1])}")

    if not all_X:
        raise RuntimeError("No windows produced. Try lowering label_purity, changing activities, or reducing window_size.")

    X = np.concatenate(all_X, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(all_y, axis=0).astype(np.int64, copy=False)
    subjects_arr = np.concatenate(all_subjects, axis=0).astype(np.int64, copy=False)

    meta: Dict[str, object] = {
        "source": "PAMAP2 Protocol",
        "protocol_dir": str(protocol_dir),
        "subjects": cfg.subjects,
        "activities": cfg.activities,
        "activity_to_label": activity_to_label,
        "include_gyro": cfg.include_gyro,
        "include_heart_rate": cfg.include_heart_rate,
        "downsample": cfg.downsample,
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "label_purity": cfg.label_purity,
        "representation": cfg.representation,
        "d": int(X.shape[1]),
        "n_windows": int(X.shape[0]),
        "n_classes": int(len(activity_to_label)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, label=y, subject=subjects_arr, meta=json.dumps(meta))
    print(f"[pamap2] wrote: {out_path}  n={int(X.shape[0])}  d={int(X.shape[1])}  C={int(len(activity_to_label))}")


if __name__ == "__main__":
    main()

