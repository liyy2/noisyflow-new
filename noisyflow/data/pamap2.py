from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class _Pamap2Table:
    x: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray


def _load_pamap2_npz(path: Path) -> _Pamap2Table:
    data = np.load(path, allow_pickle=True)
    try:
        x = np.asarray(data["X"])
        labels = np.asarray(data["label"])
        subjects = np.asarray(data["subject"])
    except KeyError as exc:
        raise KeyError(
            f"Missing key {exc!s} in {path}. Expected keys: 'X', 'label', 'subject'."
        ) from exc
    if x.ndim != 2:
        raise ValueError(f"Expected X to have shape (N,d), got {x.shape}")
    n = int(x.shape[0])
    for name, arr in {"label": labels, "subject": subjects}.items():
        if int(arr.shape[0]) != n:
            raise ValueError(f"Expected {name} to have shape (N,), got {arr.shape} with N={n}")
    return _Pamap2Table(x=x, labels=labels, subjects=subjects)


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    uniq = np.unique(labels)
    mapping: Dict[Any, int] = {v: i for i, v in enumerate(uniq.tolist())}
    encoded = np.vectorize(mapping.__getitem__, otypes=[np.int64])(labels)
    return encoded.astype(np.int64, copy=False), uniq.tolist()


def _as_size(value: Optional[Union[int, float]], *, total: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("size parameters must be int, float, or None")
    if isinstance(value, float):
        if not (0.0 < value < 1.0):
            raise ValueError("Fractional sizes must be in (0,1)")
        return max(1, int(round(total * float(value))))
    value = int(value)
    if value <= 0:
        raise ValueError("Size parameters must be > 0 when provided")
    return value


def _subsample_indices(
    rng: np.random.Generator,
    indices: np.ndarray,
    n: Optional[int],
    *,
    stratify: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n is None or n >= int(indices.shape[0]):
        out = indices.copy()
        rng.shuffle(out)
        return out
    if stratify is None:
        out = indices.copy()
        rng.shuffle(out)
        return out[:n]

    stratify = np.asarray(stratify)
    if stratify.shape[0] != indices.shape[0]:
        raise ValueError("stratify must align with indices")

    chosen: List[int] = []
    classes = np.unique(stratify)
    per_class = max(1, int(n // max(1, classes.size)))
    for c in classes.tolist():
        idx_c = indices[stratify == c]
        if idx_c.size == 0:
            continue
        idx_c = idx_c.copy()
        rng.shuffle(idx_c)
        chosen.extend(idx_c[: min(per_class, int(idx_c.size))].tolist())

    if len(chosen) < n:
        remaining = np.setdiff1d(indices, np.array(chosen, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, n - len(chosen))].tolist())

    out = np.array(chosen[:n], dtype=np.int64)
    rng.shuffle(out)
    return out


def _maybe_preprocess(
    x_train: np.ndarray,
    x_all: Sequence[np.ndarray],
    *,
    standardize: bool,
    pca_dim: Optional[int],
    seed: int,
) -> List[np.ndarray]:
    if not standardize and pca_dim is None:
        return [np.asarray(x, dtype=np.float32) for x in x_all]

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for standardize=True and/or pca_dim. "
            "Install scikit-learn or disable these options."
        ) from exc

    x_fit = np.asarray(x_train, dtype=np.float32)
    transforms: List[Any] = []

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(x_fit)
        transforms.append(scaler)
        x_fit = scaler.transform(x_fit).astype(np.float32, copy=False)

    if pca_dim is not None:
        pca_dim = int(pca_dim)
        if pca_dim <= 0:
            raise ValueError("pca_dim must be > 0")
        pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=int(seed))
        pca.fit(x_fit)
        transforms.append(pca)

    out: List[np.ndarray] = []
    for x in x_all:
        z = np.asarray(x, dtype=np.float32)
        for t in transforms:
            z = t.transform(z).astype(np.float32, copy=False)
        out.append(z)
    return out


def make_federated_pamap2(
    *,
    path: str,
    target_subject: int,
    source_subjects: Optional[Sequence[int]] = None,
    n_per_client: Optional[Union[int, float]] = None,
    target_ref_size: Optional[Union[int, float]] = 2000,
    target_test_size: Optional[Union[int, float]] = 5000,
    min_per_client: int = 1,
    standardize: bool = True,
    pca_dim: Optional[int] = None,
    preprocess_fit: str = "source_target_ref",
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Build NoisyFlow-style federated datasets from a preprocessed PAMAP2 window table.

    Input `.npz` must contain:
      - `X` (N,d): window-level feature vectors (e.g., from scripts/prepare_pamap2.py)
      - `label` (N,): integer activity labels (any encoding; re-encoded to [0..C-1])
      - `subject` (N,): subject ids (client/domain ids)

    Split:
      - Each source subject becomes a federated client dataset.
      - All samples for `target_subject` are split iid into `target_ref` and `target_test`.
    """
    rng = np.random.default_rng(int(seed))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    table = _load_pamap2_npz(path_obj)
    x = np.asarray(table.x, dtype=np.float32)
    labels_raw = np.asarray(table.labels)
    subjects = np.asarray(table.subjects, dtype=np.int64)

    labels, _ = _encode_labels(labels_raw)

    target_subject = int(target_subject)
    subject_ids = np.unique(subjects).astype(np.int64, copy=False).tolist()
    if target_subject not in set(map(int, subject_ids)):
        raise ValueError(f"target_subject={target_subject} not present in dataset subjects={sorted(map(int, subject_ids))}")

    if source_subjects is None:
        source_subject_ids = [int(s) for s in subject_ids if int(s) != target_subject]
    else:
        source_subject_ids = [int(s) for s in source_subjects]
        overlap = set(source_subject_ids).intersection({target_subject})
        if overlap:
            raise ValueError(f"source_subjects and target_subject overlap: {sorted(overlap)}")

    if not source_subject_ids:
        raise ValueError("No source subjects selected; set source_subjects or choose a different target_subject.")

    target_idx_all = np.flatnonzero(subjects == target_subject)
    if int(target_idx_all.shape[0]) < 2:
        raise ValueError(f"Not enough target samples for subject {target_subject} (n={int(target_idx_all.shape[0])}).")

    target_ref_n = _as_size(target_ref_size, total=int(target_idx_all.shape[0]))
    target_test_n = _as_size(target_test_size, total=int(target_idx_all.shape[0]))
    if target_ref_n is None:
        target_ref_n = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))
    if target_test_n is None:
        target_test_n = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))
    if target_ref_n + target_test_n > int(target_idx_all.shape[0]):
        target_test_n = min(target_test_n, int(target_idx_all.shape[0]) - 1)
        target_ref_n = min(target_ref_n, int(target_idx_all.shape[0]) - target_test_n)

    target_perm = target_idx_all.copy()
    rng.shuffle(target_perm)
    target_test_idx = target_perm[:target_test_n]
    target_ref_idx = target_perm[target_test_n : target_test_n + target_ref_n]

    target_ref_idx = _subsample_indices(rng, target_ref_idx, target_ref_n, stratify=labels[target_ref_idx])
    target_test_idx = _subsample_indices(rng, target_test_idx, target_test_n, stratify=labels[target_test_idx])

    x_target_ref = x[target_ref_idx]
    y_target_ref = labels[target_ref_idx]
    x_target_test = x[target_test_idx]
    y_target_test = labels[target_test_idx]

    client_datasets: List[TensorDataset] = []
    client_x_blocks: List[np.ndarray] = []
    client_y_blocks: List[np.ndarray] = []
    for sid in source_subject_ids:
        idx = np.flatnonzero(subjects == int(sid))
        if int(idx.shape[0]) < int(min_per_client):
            continue
        n_client = _as_size(n_per_client, total=int(idx.shape[0]))
        idx = _subsample_indices(rng, idx, n_client, stratify=labels[idx])
        x_c = x[idx]
        y_c = labels[idx]
        client_x_blocks.append(x_c)
        client_y_blocks.append(y_c)
        client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))

    if not client_datasets:
        raise ValueError("No clients available after filtering; adjust min_per_client/source_subjects.")

    # Optional global preprocessing.
    if standardize or pca_dim is not None:
        preprocess_fit = str(preprocess_fit).strip().lower()
        if preprocess_fit == "source_target_ref":
            x_fit = np.concatenate([np.concatenate(client_x_blocks, axis=0), x_target_ref], axis=0)
        elif preprocess_fit == "source_only":
            x_fit = np.concatenate(client_x_blocks, axis=0)
        else:
            raise ValueError("preprocess_fit must be 'source_target_ref' or 'source_only'")
        x_all = list(client_x_blocks) + [x_target_ref, x_target_test]
        x_all_proc = _maybe_preprocess(
            x_fit,
            x_all,
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )
        client_x_blocks = x_all_proc[: len(client_x_blocks)]
        x_target_ref = x_all_proc[-2]
        x_target_test = x_all_proc[-1]

        client_datasets = [
            TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c))
            for x_c, y_c in zip(client_x_blocks, client_y_blocks)
        ]

    target_ref = TensorDataset(torch.from_numpy(x_target_ref), torch.from_numpy(y_target_ref))
    target_test = TensorDataset(torch.from_numpy(x_target_test), torch.from_numpy(y_target_test))
    return client_datasets, target_ref, target_test
