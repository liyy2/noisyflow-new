from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset

from noisyflow.data.cell import _as_size, _maybe_preprocess


def _load_h5ad_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import anndata
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Loading .h5ad requires anndata (and h5py). Install with `pip install anndata h5py`."
        ) from exc

    adata = anndata.read_h5ad(path)
    if "drug" not in adata.obs.columns:
        raise KeyError(f"Missing obs['drug'] in {path}. Available: {list(adata.obs.columns)}")
    x = np.asarray(adata.X)
    if x.ndim != 2:
        raise ValueError(f"Expected adata.X to have shape (N,d), got {x.shape}")
    drugs = np.asarray(adata.obs["drug"])
    if drugs.shape[0] != x.shape[0]:
        raise ValueError(f"Expected obs['drug'] length {x.shape[0]}, got {drugs.shape[0]}")
    return x, drugs


def _cluster_labels(
    x_fit: np.ndarray,
    x_blocks: List[np.ndarray],
    *,
    num_labels: int,
    seed: int,
) -> List[np.ndarray]:
    if num_labels <= 1:
        return [np.zeros((int(x.shape[0]),), dtype=np.int64) for x in x_blocks]
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("label_mode='kmeans' requires scikit-learn (pip install scikit-learn).") from exc

    n_fit = int(x_fit.shape[0])
    k = min(int(num_labels), max(1, n_fit))
    if k <= 1:
        return [np.zeros((int(x.shape[0]),), dtype=np.int64) for x in x_blocks]

    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=2048,
        n_init="auto",
        random_state=int(seed),
    )
    km.fit(np.asarray(x_fit, dtype=np.float32))
    return [km.predict(np.asarray(x, dtype=np.float32)).astype(np.int64, copy=False) for x in x_blocks]


def make_federated_4i_proteomics(
    *,
    path: str = "datasets/4i/8h.h5ad",
    source_drug: str = "control",
    target_drug: str = "dasatinib",
    n_source_clients: int = 5,
    source_size_per_client: Optional[Union[int, float]] = None,
    target_ref_size: Optional[Union[int, float]] = None,
    target_test_size: Optional[Union[int, float]] = None,
    min_cells_per_client: int = 1,
    standardize: bool = True,
    pca_dim: Optional[int] = 32,
    label_mode: str = "kmeans",
    num_labels: int = 10,
    seed: int = 0,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset]:
    """
    Build NoisyFlow-style federated datasets from a 4i single-cell proteomics table.

    This dataset ships as an `.h5ad` file with:
      - `adata.X`: (N, d) protein/morphology features
      - `adata.obs['drug']`: drug condition per cell

    Splits:
      - Source/private: all cells with `drug == source_drug`, partitioned into `n_source_clients` pseudo-clients.
      - Target: cells with `drug == target_drug`, split iid into `target_ref` and `target_test`.

    Labels:
      - `label_mode='kmeans'` (default): fit MiniBatchKMeans on (source + target_ref) and assign cluster IDs.
      - `label_mode='none'`: all labels are 0 (single-class; use distribution metrics only).
    """
    rng = np.random.default_rng(int(seed))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    x_raw, drugs = _load_h5ad_matrix(path_obj)
    x_raw = np.asarray(x_raw, dtype=np.float32)

    source_mask = drugs == source_drug
    target_mask = drugs == target_drug
    if not np.any(source_mask):
        raise ValueError(f"No rows found for source_drug='{source_drug}'")
    if not np.any(target_mask):
        raise ValueError(f"No rows found for target_drug='{target_drug}'")

    source_idx = np.flatnonzero(source_mask)
    target_idx_all = np.flatnonzero(target_mask)

    n_test = _as_size(target_test_size, total=int(target_idx_all.shape[0]))
    if n_test is None:
        n_test = max(1, int(round(0.2 * float(target_idx_all.shape[0]))))
    target_perm = target_idx_all.copy()
    rng.shuffle(target_perm)
    target_test_idx = target_perm[:n_test]
    target_ref_idx = target_perm[n_test:]

    target_ref_n = _as_size(target_ref_size, total=int(target_ref_idx.shape[0]))
    if target_ref_n is not None and target_ref_n < int(target_ref_idx.shape[0]):
        rng.shuffle(target_ref_idx)
        target_ref_idx = target_ref_idx[:target_ref_n]

    if int(target_ref_idx.shape[0]) == 0 or int(target_test_idx.shape[0]) == 0:
        raise ValueError("Empty target_ref or target_test split; adjust target_ref_size/target_test_size.")

    x_source_all = x_raw[source_idx]
    x_target_ref = x_raw[target_ref_idx]
    x_target_test = x_raw[target_test_idx]

    if standardize or pca_dim is not None:
        x_fit = np.concatenate([x_source_all, x_target_ref], axis=0)
        x_source_all, x_target_ref, x_target_test = _maybe_preprocess(
            x_fit,
            [x_source_all, x_target_ref, x_target_test],
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )

    label_mode = str(label_mode).strip().lower()
    if label_mode not in {"kmeans", "none"}:
        raise ValueError("label_mode must be 'kmeans' or 'none'")

    if label_mode == "none":
        y_source_all = np.zeros((int(x_source_all.shape[0]),), dtype=np.int64)
        y_target_ref = np.zeros((int(x_target_ref.shape[0]),), dtype=np.int64)
        y_target_test = np.zeros((int(x_target_test.shape[0]),), dtype=np.int64)
    else:
        y_source_all, y_target_ref, y_target_test = _cluster_labels(
            np.concatenate([x_source_all, x_target_ref], axis=0),
            [x_source_all, x_target_ref, x_target_test],
            num_labels=int(num_labels),
            seed=seed,
        )

    perm = rng.permutation(int(x_source_all.shape[0]))
    x_source_all = x_source_all[perm]
    y_source_all = y_source_all[perm]

    n_source_clients = int(n_source_clients)
    if n_source_clients <= 0:
        raise ValueError("n_source_clients must be > 0")

    x_splits = np.array_split(x_source_all, n_source_clients)
    y_splits = np.array_split(y_source_all, n_source_clients)
    client_datasets: List[TensorDataset] = []
    for x_c, y_c in zip(x_splits, y_splits):
        if int(x_c.shape[0]) < int(min_cells_per_client):
            continue
        n_keep = _as_size(source_size_per_client, total=int(x_c.shape[0]))
        if n_keep is not None and n_keep < int(x_c.shape[0]):
            idx = rng.permutation(int(x_c.shape[0]))[:n_keep]
            x_c = x_c[idx]
            y_c = y_c[idx]
        client_datasets.append(TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c)))

    if not client_datasets:
        raise ValueError("No clients available after filtering; check n_source_clients/min_cells_per_client.")

    target_ref = TensorDataset(torch.from_numpy(x_target_ref), torch.from_numpy(y_target_ref))
    target_test = TensorDataset(torch.from_numpy(x_target_test), torch.from_numpy(y_target_test))
    return client_datasets, target_ref, target_test


def list_4i_drugs(*, path: str = "datasets/4i/8h.h5ad") -> List[str]:
    """
    Return sorted unique drug labels available in the 4i dataset.
    """
    x, drugs = _load_h5ad_matrix(Path(path))
    _ = x  # unused
    return sorted(np.unique(drugs).tolist())


def summarize_4i_dataset(*, path: str = "datasets/4i/8h.h5ad") -> dict[str, Any]:
    """
    Lightweight dataset summary for sanity checks and config selection.
    """
    x, drugs = _load_h5ad_matrix(Path(path))
    uniq, counts = np.unique(drugs, return_counts=True)
    return {
        "path": str(path),
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "num_drugs": int(uniq.shape[0]),
        "drugs": {str(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())},
    }

