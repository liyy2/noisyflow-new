from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


CAMELYON17_SPLIT_DICT: Dict[str, int] = {"train": 0, "id_val": 1, "test": 2, "val": 3}


@dataclass(frozen=True)
class _Camelyon17Table:
    x: np.ndarray
    labels: np.ndarray
    hospitals: np.ndarray
    splits: np.ndarray


def _load_camelyon17_npz(path: Path) -> _Camelyon17Table:
    data = np.load(path, allow_pickle=True)
    try:
        x = np.asarray(data["X"])
        labels = np.asarray(data["label"])
        hospitals = np.asarray(data["hospital"])
        splits = np.asarray(data["split"])
    except KeyError as exc:
        raise KeyError(
            f"Missing key {exc!s} in {path}. Expected keys: 'X', 'label', 'hospital', 'split'."
        ) from exc
    if x.ndim != 2:
        raise ValueError(f"Expected X to have shape (N,d), got {x.shape}")
    n = int(x.shape[0])
    for name, arr in {"label": labels, "hospital": hospitals, "split": splits}.items():
        if int(arr.shape[0]) != n:
            raise ValueError(f"Expected {name} to have shape (N,), got {arr.shape} with N={n}")
    return _Camelyon17Table(x=x, labels=labels, hospitals=hospitals, splits=splits)


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


def _normalize_splits(splits: Sequence[Union[str, int]]) -> List[int]:
    out: List[int] = []
    for s in splits:
        if isinstance(s, str):
            key = s.strip()
            if key not in CAMELYON17_SPLIT_DICT:
                raise ValueError(f"Unknown split '{s}'. Valid: {sorted(CAMELYON17_SPLIT_DICT.keys())}")
            out.append(int(CAMELYON17_SPLIT_DICT[key]))
        else:
            out.append(int(s))
    return out


def make_federated_camelyon17_wilds(
    *,
    path: str,
    source_splits: Sequence[Union[str, int]] = ("train", "id_val"),
    target_split: Union[str, int] = "test",
    source_hospitals: Optional[Sequence[int]] = None,
    target_hospital: Optional[int] = 2,
    target_hospitals: Optional[Sequence[int]] = None,
    n_per_client: Optional[Union[int, float]] = 20000,
    target_ref_size: Optional[Union[int, float]] = 2000,
    target_test_size: Optional[Union[int, float]] = 10000,
    min_per_client: int = 1,
    standardize: bool = False,
    pca_dim: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Build NoisyFlow-style federated datasets from CAMELYON17-WILDS embedding features.

    Input:
      - `.npz` produced by `scripts/prepare_camelyon17_wilds.py`, with keys:
        `X` (N,d), `label` (N,), `hospital` (N,), `split` (N,)

    Splits:
      - Source clients are formed by hospital id from `source_splits`.
      - Target samples are from `target_split` filtered by either `target_hospital` (single domain) or
        `target_hospitals` (multi-domain) and split iid into `target_ref` / `target_test`.
    """
    rng = np.random.default_rng(int(seed))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    table = _load_camelyon17_npz(path_obj)
    x = np.asarray(table.x, dtype=np.float32)
    labels_raw = np.asarray(table.labels)
    hospitals = np.asarray(table.hospitals)
    splits = np.asarray(table.splits, dtype=np.int64)

    encoded_labels, _ = _encode_labels(labels_raw)

    source_split_ids = set(_normalize_splits(list(source_splits)))
    target_split_id = _normalize_splits([target_split])[0]

    target_hospital_ids: Optional[List[int]] = None
    if target_hospitals is not None:
        target_hospital_ids = [int(h) for h in target_hospitals]
        if not target_hospital_ids:
            raise ValueError("target_hospitals must be non-empty when provided.")
    elif target_hospital is not None:
        target_hospital_ids = [int(target_hospital)]

    if source_hospitals is not None and target_hospital_ids is not None:
        overlap = set(map(int, source_hospitals)).intersection(set(target_hospital_ids))
        if overlap:
            raise ValueError(
                "source_hospitals and target_hospital(s) overlap: "
                f"{sorted(overlap)}. Use disjoint source/target hospital sets."
            )

    source_mask = np.isin(splits, np.array(sorted(source_split_ids), dtype=np.int64))
    if source_hospitals is not None:
        source_mask &= np.isin(hospitals, np.array(list(source_hospitals), dtype=hospitals.dtype))
    if target_hospital_ids is not None:
        source_mask &= ~np.isin(hospitals, np.array(target_hospital_ids, dtype=hospitals.dtype))

    target_mask = splits == int(target_split_id)
    if target_hospital_ids is not None:
        target_mask &= np.isin(hospitals, np.array(target_hospital_ids, dtype=hospitals.dtype))

    if not np.any(source_mask):
        raise ValueError("No source rows selected; check source_splits/source_hospitals/target_hospital.")
    if not np.any(target_mask):
        raise ValueError("No target rows selected; check target_split/target_hospital(s).")

    source_idx_all = np.flatnonzero(source_mask)
    target_idx_all = np.flatnonzero(target_mask)

    client_ids = np.unique(hospitals[source_idx_all]).tolist()
    client_datasets: List[TensorDataset] = []
    client_x_blocks: List[np.ndarray] = []
    for cid in client_ids:
        cid_int = int(cid)
        idx = source_idx_all[hospitals[source_idx_all] == cid_int]
        if int(idx.shape[0]) < int(min_per_client):
            continue
        n_client = _as_size(n_per_client, total=int(idx.shape[0]))
        idx = _subsample_indices(rng, idx, n_client, stratify=encoded_labels[idx])
        x_c = x[idx]
        y_c = encoded_labels[idx]
        client_x_blocks.append(x_c)
        client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))

    if not client_datasets:
        raise ValueError("No clients available after filtering; adjust min_per_client/source filters.")

    target_ref_n = _as_size(target_ref_size, total=int(target_idx_all.shape[0]))
    target_test_n = _as_size(target_test_size, total=int(target_idx_all.shape[0]))
    if target_ref_n is None:
        target_ref_n = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))
    if target_test_n is None:
        target_test_n = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))

    if target_ref_n + target_test_n > int(target_idx_all.shape[0]):
        # Prioritize test, then ref.
        target_test_n = min(target_test_n, int(target_idx_all.shape[0]) - 1)
        target_ref_n = min(target_ref_n, int(target_idx_all.shape[0]) - target_test_n)

    target_perm = target_idx_all.copy()
    rng.shuffle(target_perm)
    target_test_idx = target_perm[:target_test_n]
    target_ref_idx = target_perm[target_test_n : target_test_n + target_ref_n]

    target_ref_idx = _subsample_indices(rng, target_ref_idx, target_ref_n, stratify=encoded_labels[target_ref_idx])
    target_test_idx = _subsample_indices(rng, target_test_idx, target_test_n, stratify=encoded_labels[target_test_idx])

    if int(target_ref_idx.shape[0]) == 0 or int(target_test_idx.shape[0]) == 0:
        raise ValueError("Empty target_ref or target_test split; adjust target_*_size parameters.")

    x_target_ref = x[target_ref_idx]
    y_target_ref = encoded_labels[target_ref_idx]
    x_target_test = x[target_test_idx]
    y_target_test = encoded_labels[target_test_idx]

    # Optional global preprocessing (fit on source+target_ref, apply to all splits).
    if standardize or pca_dim is not None:
        x_fit = np.concatenate([np.concatenate(client_x_blocks, axis=0), x_target_ref], axis=0)
        client_x_blocks_proc, x_target_ref_proc, x_target_test_proc = _maybe_preprocess(
            x_fit,
            [np.concatenate(client_x_blocks, axis=0), x_target_ref, x_target_test],
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )
        x_all_clients_proc = client_x_blocks_proc

        offsets = np.cumsum([0] + [int(ds.tensors[0].shape[0]) for ds in client_datasets])
        new_client_datasets: List[TensorDataset] = []
        for i, ds in enumerate(client_datasets):
            a = int(offsets[i])
            b = int(offsets[i + 1])
            x_c = x_all_clients_proc[a:b]
            y_c = ds.tensors[1].numpy().astype(np.int64, copy=False)
            new_client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))
        client_datasets = new_client_datasets
        x_target_ref = x_target_ref_proc
        x_target_test = x_target_test_proc

    target_ref = TensorDataset(torch.from_numpy(x_target_ref), torch.from_numpy(y_target_ref))
    target_test = TensorDataset(torch.from_numpy(x_target_test), torch.from_numpy(y_target_test))
    return client_datasets, target_ref, target_test


def make_federated_camelyon17(
    *,
    path: str,
    source_hospitals: Sequence[int],
    target_hospitals: Sequence[int],
    source_splits: Optional[Sequence[Union[str, int]]] = None,
    target_splits: Optional[Sequence[Union[str, int]]] = None,
    n_per_client: Optional[Union[int, float]] = 20000,
    target_ref_size: Optional[Union[int, float]] = 2000,
    target_test_size: Optional[Union[int, float]] = 10000,
    min_per_client: int = 1,
    standardize: bool = False,
    pca_dim: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Build NoisyFlow-style federated datasets from CAMELYON17 embedding features.

    This builder defines source/target domains explicitly via hospital id lists. Optionally,
    restrict each side to specific WILDS split ids/names via `source_splits` / `target_splits`.

    Input:
      - `.npz` with keys: `X` (N,d), `label` (N,), `hospital` (N,), `split` (N,)
    """
    rng = np.random.default_rng(int(seed))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    source_hosp_ids = [int(h) for h in source_hospitals]
    target_hosp_ids = [int(h) for h in target_hospitals]
    if not source_hosp_ids:
        raise ValueError("source_hospitals must be non-empty")
    if not target_hosp_ids:
        raise ValueError("target_hospitals must be non-empty")

    overlap = set(source_hosp_ids).intersection(set(target_hosp_ids))
    if overlap:
        raise ValueError(f"source_hospitals and target_hospitals must be disjoint; overlap={sorted(overlap)}")

    table = _load_camelyon17_npz(path_obj)
    x = np.asarray(table.x, dtype=np.float32)
    labels_raw = np.asarray(table.labels)
    hospitals = np.asarray(table.hospitals)
    splits = np.asarray(table.splits, dtype=np.int64)

    encoded_labels, _ = _encode_labels(labels_raw)

    source_mask = np.isin(hospitals, np.array(source_hosp_ids, dtype=hospitals.dtype))
    if source_splits is not None:
        source_split_ids = set(_normalize_splits(list(source_splits)))
        source_mask &= np.isin(splits, np.array(sorted(source_split_ids), dtype=np.int64))

    target_mask = np.isin(hospitals, np.array(target_hosp_ids, dtype=hospitals.dtype))
    if target_splits is not None:
        target_split_ids = set(_normalize_splits(list(target_splits)))
        target_mask &= np.isin(splits, np.array(sorted(target_split_ids), dtype=np.int64))

    if not np.any(source_mask):
        raise ValueError("No source rows selected; check source_hospitals/source_splits.")
    if not np.any(target_mask):
        raise ValueError("No target rows selected; check target_hospitals/target_splits.")

    source_idx_all = np.flatnonzero(source_mask)
    target_idx_all = np.flatnonzero(target_mask)

    client_datasets: List[TensorDataset] = []
    client_x_blocks: List[np.ndarray] = []
    for cid in source_hosp_ids:
        idx = source_idx_all[hospitals[source_idx_all] == int(cid)]
        if int(idx.shape[0]) < int(min_per_client):
            continue
        n_client = _as_size(n_per_client, total=int(idx.shape[0]))
        idx = _subsample_indices(rng, idx, n_client, stratify=encoded_labels[idx])
        x_c = x[idx]
        y_c = encoded_labels[idx]
        client_x_blocks.append(x_c)
        client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))

    if not client_datasets:
        raise ValueError("No clients available after filtering; adjust min_per_client/source filters.")

    target_ref_n = _as_size(target_ref_size, total=int(target_idx_all.shape[0]))
    target_test_n = _as_size(target_test_size, total=int(target_idx_all.shape[0]))
    if target_ref_n is None:
        target_ref_n = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))
    if target_test_n is None:
        target_test_n = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))

    if target_ref_n + target_test_n > int(target_idx_all.shape[0]):
        # Prioritize test, then ref.
        target_test_n = min(target_test_n, int(target_idx_all.shape[0]) - 1)
        target_ref_n = min(target_ref_n, int(target_idx_all.shape[0]) - target_test_n)

    target_perm = target_idx_all.copy()
    rng.shuffle(target_perm)
    target_test_idx = target_perm[:target_test_n]
    target_ref_idx = target_perm[target_test_n : target_test_n + target_ref_n]

    target_ref_idx = _subsample_indices(rng, target_ref_idx, target_ref_n, stratify=encoded_labels[target_ref_idx])
    target_test_idx = _subsample_indices(rng, target_test_idx, target_test_n, stratify=encoded_labels[target_test_idx])

    if int(target_ref_idx.shape[0]) == 0 or int(target_test_idx.shape[0]) == 0:
        raise ValueError("Empty target_ref or target_test split; adjust target_*_size parameters.")

    x_target_ref = x[target_ref_idx]
    y_target_ref = encoded_labels[target_ref_idx]
    x_target_test = x[target_test_idx]
    y_target_test = encoded_labels[target_test_idx]

    if standardize or pca_dim is not None:
        x_fit = np.concatenate([np.concatenate(client_x_blocks, axis=0), x_target_ref], axis=0)
        client_x_blocks_proc, x_target_ref_proc, x_target_test_proc = _maybe_preprocess(
            x_fit,
            [np.concatenate(client_x_blocks, axis=0), x_target_ref, x_target_test],
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )
        x_all_clients_proc = client_x_blocks_proc

        offsets = np.cumsum([0] + [int(ds.tensors[0].shape[0]) for ds in client_datasets])
        new_client_datasets: List[TensorDataset] = []
        for i, ds in enumerate(client_datasets):
            a = int(offsets[i])
            b = int(offsets[i + 1])
            x_c = x_all_clients_proc[a:b]
            y_c = ds.tensors[1].numpy().astype(np.int64, copy=False)
            new_client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))
        client_datasets = new_client_datasets
        x_target_ref = x_target_ref_proc
        x_target_test = x_target_test_proc

    target_ref = TensorDataset(torch.from_numpy(x_target_ref), torch.from_numpy(y_target_ref))
    target_test = TensorDataset(torch.from_numpy(x_target_test), torch.from_numpy(y_target_test))
    return client_datasets, target_ref, target_test
