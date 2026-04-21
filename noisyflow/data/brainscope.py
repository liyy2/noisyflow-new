from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class _BrainScopeTable:
    x: np.ndarray
    cohort: np.ndarray
    disorder: Optional[np.ndarray]
    labels_case_control: Optional[np.ndarray]
    labels_neurodegenerative: Optional[np.ndarray]


def _load_brainscope_npz(path: Path) -> _BrainScopeTable:
    data = np.load(path, allow_pickle=True)
    try:
        x = np.asarray(data["X"])
        cohort = np.asarray(data["cohort"])
    except KeyError as exc:
        raise KeyError(f"Missing key {exc!s} in {path}. Expected keys include 'X' and 'cohort'.") from exc

    disorder = np.asarray(data["disorder"]) if "disorder" in data else None
    labels_case_control = np.asarray(data["label_case_control"]) if "label_case_control" in data else None
    labels_neurodegenerative = (
        np.asarray(data["label_neurodegenerative"]) if "label_neurodegenerative" in data else None
    )

    if x.ndim != 2:
        raise ValueError(f"Expected X to have shape (N,d), got {x.shape}")
    n = int(x.shape[0])
    if cohort.shape[0] != n:
        raise ValueError(f"Expected cohort to have shape (N,), got {cohort.shape} with N={n}")
    for name, arr in {
        "disorder": disorder,
        "label_case_control": labels_case_control,
        "label_neurodegenerative": labels_neurodegenerative,
    }.items():
        if arr is not None and int(arr.shape[0]) != n:
            raise ValueError(f"Expected {name} to have shape (N,), got {arr.shape} with N={n}")

    return _BrainScopeTable(
        x=x,
        cohort=cohort,
        disorder=disorder,
        labels_case_control=labels_case_control,
        labels_neurodegenerative=labels_neurodegenerative,
    )


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


def _normalize_str_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    out = [str(v).strip() for v in values]
    out = [v for v in out if v]
    return out or None


def make_federated_brainscope(
    *,
    path: str,
    label_mode: str = "case_control",
    source_cohorts: Sequence[str] = ("CMC",),
    target_cohorts: Optional[Sequence[str]] = None,
    include_disorders: Optional[Sequence[str]] = None,
    exclude_disorders: Optional[Sequence[str]] = None,
    n_per_client: Optional[Union[int, float]] = None,
    target_ref_size: Optional[Union[int, float]] = None,
    target_test_size: Optional[Union[int, float]] = None,
    min_per_client: int = 1,
    standardize: bool = False,
    pca_dim: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Build NoisyFlow-style federated datasets from a BrainSCOPE-like cohort table.

    Input:
      - `.npz` produced by `scripts/prepare_brainscope_aging_yl.py`, with keys:
        `X` (N,G), `cohort` (N,), `disorder` (N,), plus label arrays.

    Setup:
      - Source clients are cohorts listed in `source_cohorts` (each cohort becomes one client).
      - Target samples are cohorts in `target_cohorts` (or all non-source cohorts) and split iid into
        `target_ref` / `target_test`.

    Labels:
      - `label_mode='case_control'`: 1 if disorder != control, else 0.
      - `label_mode='neurodegenerative'`: 1 if disorder in {Alzheimers/dementia, cognitive impairment}, else 0.
    """
    rng = np.random.default_rng(int(seed))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/prepare_brainscope_aging_yl.py first.")

    table = _load_brainscope_npz(path_obj)
    x = np.asarray(table.x, dtype=np.float32)
    cohort = np.asarray(table.cohort).astype(str)
    disorder = np.asarray(table.disorder).astype(str) if table.disorder is not None else None

    label_mode = str(label_mode).strip().lower()
    if label_mode == "case_control":
        if table.labels_case_control is None:
            raise KeyError("label_case_control missing from npz; regenerate with scripts/prepare_brainscope_aging_yl.py")
        labels = np.asarray(table.labels_case_control, dtype=np.int64)
    elif label_mode in {"neurodegenerative", "alzheimers"}:
        if table.labels_neurodegenerative is None:
            raise KeyError(
                "label_neurodegenerative missing from npz; regenerate with scripts/prepare_brainscope_aging_yl.py"
            )
        labels = np.asarray(table.labels_neurodegenerative, dtype=np.int64)
    else:
        raise ValueError("label_mode must be 'case_control' or 'neurodegenerative'")

    mask = np.ones((int(x.shape[0]),), dtype=bool)
    include_disorders = _normalize_str_list(include_disorders)
    exclude_disorders = _normalize_str_list(exclude_disorders)
    if include_disorders is not None:
        if disorder is None:
            raise ValueError("include_disorders requires disorder field in the .npz")
        mask &= np.isin(disorder, np.array(include_disorders, dtype=object))
    if exclude_disorders is not None:
        if disorder is None:
            raise ValueError("exclude_disorders requires disorder field in the .npz")
        mask &= ~np.isin(disorder, np.array(exclude_disorders, dtype=object))

    if not np.any(mask):
        raise ValueError("No samples left after disorder filtering; adjust include_disorders/exclude_disorders.")

    x = x[mask]
    cohort = cohort[mask]
    labels = labels[mask]

    source_cohorts = _normalize_str_list(source_cohorts) or []
    if not source_cohorts:
        raise ValueError("source_cohorts must be non-empty")

    if target_cohorts is None:
        target_mask = ~np.isin(cohort, np.array(source_cohorts, dtype=object))
    else:
        target_cohorts_norm = _normalize_str_list(target_cohorts) or []
        if not target_cohorts_norm:
            raise ValueError("target_cohorts must be non-empty when provided")
        target_mask = np.isin(cohort, np.array(target_cohorts_norm, dtype=object))

    source_mask = np.isin(cohort, np.array(source_cohorts, dtype=object))
    if not np.any(source_mask):
        raise ValueError(f"No source rows selected for source_cohorts={source_cohorts}")
    if not np.any(target_mask):
        raise ValueError("No target rows selected; check target_cohorts/source_cohorts and disorder filters.")

    source_idx_all = np.flatnonzero(source_mask)
    target_idx_all = np.flatnonzero(target_mask)

    # Build source clients by cohort id.
    client_datasets: List[TensorDataset] = []
    client_x_blocks: List[np.ndarray] = []
    for cohort_id in source_cohorts:
        idx = source_idx_all[cohort[source_idx_all] == cohort_id]
        if int(idx.shape[0]) < int(min_per_client):
            continue
        n_client = _as_size(n_per_client, total=int(idx.shape[0]))
        idx = _subsample_indices(rng, idx, n_client, stratify=labels[idx])
        x_c = x[idx]
        y_c = labels[idx]
        client_x_blocks.append(x_c)
        client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))

    if not client_datasets:
        raise ValueError("No clients available after filtering; check source_cohorts/min_per_client/n_per_client.")

    # Target split iid into ref/test.
    n_target = int(target_idx_all.shape[0])
    n_test = _as_size(target_test_size, total=n_target)
    if n_test is None:
        n_test = max(1, int(round(0.2 * float(n_target))))
    n_test = min(n_test, n_target - 1) if n_target > 1 else n_test

    target_perm = target_idx_all.copy()
    rng.shuffle(target_perm)
    target_test_idx = target_perm[:n_test]
    target_ref_idx = target_perm[n_test:]

    target_ref_n = _as_size(target_ref_size, total=int(target_ref_idx.shape[0]))
    if target_ref_n is not None:
        target_ref_idx = _subsample_indices(rng, target_ref_idx, target_ref_n, stratify=labels[target_ref_idx])

    if int(target_ref_idx.shape[0]) == 0 or int(target_test_idx.shape[0]) == 0:
        raise ValueError("Empty target_ref or target_test split; adjust target_ref_size/target_test_size.")

    x_target_ref = x[target_ref_idx]
    y_target_ref = labels[target_ref_idx]
    x_target_test = x[target_test_idx]
    y_target_test = labels[target_test_idx]

    # Optional global preprocessing (fit on source+target_ref, apply to all splits).
    if standardize or pca_dim is not None:
        x_fit = np.concatenate([np.concatenate(client_x_blocks, axis=0), x_target_ref], axis=0)
        x_clients_flat, x_ref_proc, x_test_proc = _maybe_preprocess(
            x_fit,
            [np.concatenate(client_x_blocks, axis=0), x_target_ref, x_target_test],
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )

        # Re-split client flat block back into per-client arrays.
        offsets = np.cumsum([0] + [int(ds.tensors[0].shape[0]) for ds in client_datasets])
        new_client_datasets: List[TensorDataset] = []
        for i, ds in enumerate(client_datasets):
            a = int(offsets[i])
            b = int(offsets[i + 1])
            x_c = x_clients_flat[a:b]
            y_c = ds.tensors[1].numpy().astype(np.int64, copy=False)
            new_client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))
        client_datasets = new_client_datasets
        x_target_ref = x_ref_proc
        x_target_test = x_test_proc

    target_ref = TensorDataset(torch.from_numpy(x_target_ref), torch.from_numpy(y_target_ref))
    target_test = TensorDataset(torch.from_numpy(x_target_test), torch.from_numpy(y_target_test))
    return client_datasets, target_ref, target_test

