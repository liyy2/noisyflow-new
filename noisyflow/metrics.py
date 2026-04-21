from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


@torch.no_grad()
def _subsample_rows(x: torch.Tensor, max_rows: int, *, seed: int = 0) -> torch.Tensor:
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0")
    n = int(x.shape[0])
    if n <= max_rows:
        return x
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen, device=x.device)[:max_rows]
    return x.index_select(0, idx)


@torch.no_grad()
def sliced_w2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_projections: int = 128,
    max_samples: Optional[int] = 2000,
    seed: int = 0,
) -> float:
    """
    Approximate W2 via the Sliced Wasserstein-2 distance.

    Definition: SW2^2(x,y) = E_u[ W2^2(<u,x>, <u,y>) ] where u is a random unit direction.
    We estimate the expectation with `num_projections` random projections and use the closed-form
    1D W2^2 estimator based on sorted samples.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors of shape (N,d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dim mismatch: {x.shape[1]} vs {y.shape[1]}")
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("x and y must be non-empty")
    num_projections = int(num_projections)
    if num_projections <= 0:
        raise ValueError("num_projections must be > 0")

    x = x.float()
    y = y.float()

    if max_samples is not None:
        x = _subsample_rows(x, int(max_samples), seed=seed)
        y = _subsample_rows(y, int(max_samples), seed=seed + 1)

    n = int(min(x.shape[0], y.shape[0]))
    if n <= 1:
        return float("nan")
    if int(x.shape[0]) != n:
        x = x[:n]
    if int(y.shape[0]) != n:
        y = y[:n]

    d = int(x.shape[1])
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    proj = torch.randn(d, num_projections, generator=gen, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min_(1e-12)

    x_proj = x @ proj  # (n, P)
    y_proj = y @ proj  # (n, P)
    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)
    w2_sq_per_proj = (x_sorted - y_sorted).pow(2).mean(dim=0)  # (P,)
    sw2_sq = w2_sq_per_proj.mean()
    return float(torch.sqrt(sw2_sq).cpu().item())


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, *, gamma: float) -> torch.Tensor:
    """
    Squared MMD with an RBF kernel k(a,b)=exp(-gamma*||a-b||^2).

    Returns a scalar tensor on the same device as inputs.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors of shape (N,d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dim mismatch: {x.shape[1]} vs {y.shape[1]}")
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("x and y must be non-empty")
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    x = x.float()
    y = y.float()

    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True)
    d_xx = (x_norm + x_norm.t() - 2.0 * (x @ x.t())).clamp_min_(0.0)
    d_yy = (y_norm + y_norm.t() - 2.0 * (y @ y.t())).clamp_min_(0.0)
    d_xy = (x_norm + y_norm.t() - 2.0 * (x @ y.t())).clamp_min_(0.0)

    k_xx = torch.exp(-gamma * d_xx)
    k_yy = torch.exp(-gamma * d_yy)
    k_xy = torch.exp(-gamma * d_xy)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


@torch.no_grad()
def rbf_mmd2_multi_gamma(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    gammas: Iterable[float],
    max_samples: Optional[int] = 2000,
    seed: int = 0,
) -> List[float]:
    """
    Compute RBF MMD^2 for multiple gammas, optionally subsampling rows for scalability.
    """
    if max_samples is not None:
        x = _subsample_rows(x, int(max_samples), seed=seed)
        y = _subsample_rows(y, int(max_samples), seed=seed + 1)
    out: List[float] = []
    for g in gammas:
        out.append(float(rbf_mmd2(x, y, gamma=float(g)).cpu().item()))
    return out


def _require_sklearn():
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for structure-preservation metrics.") from exc
    return silhouette_score, NearestNeighbors


def _as_numpy_2d(x, *, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _as_numpy_1d(x, *, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr).reshape(-1)
    return arr


def label_silhouette_score(x, labels) -> float:
    """
    Compute the silhouette score of features with respect to biology labels.

    Args:
        x: Feature matrix of shape `(N, d)`.
        labels: Biology labels of shape `(N,)`.

    Returns:
        The Euclidean silhouette score. Returns `nan` when the score is undefined.
    """
    silhouette_score, _ = _require_sklearn()

    x_np = _as_numpy_2d(x, name="x")
    labels_np = _as_numpy_1d(labels, name="labels")
    if x_np.shape[0] != labels_np.shape[0]:
        raise ValueError(f"x and labels must align, got {x_np.shape[0]} vs {labels_np.shape[0]}")

    uniq, counts = np.unique(labels_np, return_counts=True)
    if uniq.size < 2 or x_np.shape[0] <= uniq.size or np.any(counts < 2):
        return float("nan")
    return float(silhouette_score(x_np, labels_np, metric="euclidean"))


def same_label_domain_mixing(
    x,
    labels,
    domains,
    *,
    n_neighbors: int = 15,
) -> float:
    """
    Measure target/synthetic mixing within each biology label neighborhood.

    For each label, this computes the fraction of opposite-domain neighbors among the
    `n_neighbors` nearest neighbors (excluding self), then averages across labels.

    Args:
        x: Feature matrix of shape `(N, d)`.
        labels: Biology labels of shape `(N,)`.
        domains: Domain ids of shape `(N,)`, e.g. target=0 and synthetic=1.
        n_neighbors: Number of neighbors per point.

    Returns:
        Average opposite-domain neighbor fraction across labels. Returns `nan` when undefined.
    """
    _, NearestNeighbors = _require_sklearn()

    x_np = _as_numpy_2d(x, name="x")
    labels_np = _as_numpy_1d(labels, name="labels")
    domains_np = _as_numpy_1d(domains, name="domains")
    if not (x_np.shape[0] == labels_np.shape[0] == domains_np.shape[0]):
        raise ValueError(
            "x, labels, and domains must align, got "
            f"{x_np.shape[0]}, {labels_np.shape[0]}, {domains_np.shape[0]}"
        )

    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be > 0")

    scores: List[float] = []
    for label in np.unique(labels_np).tolist():
        mask = labels_np == label
        x_label = x_np[mask]
        domains_label = domains_np[mask]
        if x_label.shape[0] <= 1 or np.unique(domains_label).size < 2:
            continue
        k = min(n_neighbors, int(x_label.shape[0]) - 1)
        if k <= 0:
            continue
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(x_label)
        nn_idx = knn.kneighbors(return_distance=False)[:, 1:]
        nn_domains = domains_label[nn_idx]
        scores.append(float(np.mean(nn_domains != domains_label[:, None])))

    if not scores:
        return float("nan")
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def per_label_centroid_distances(
    x,
    labels,
    target_x,
    target_labels,
) -> Dict[int, float]:
    """
    Compute per-label centroid distances from a synthetic set to a target set.

    Args:
        x: Synthetic feature matrix of shape `(N, d)`.
        labels: Synthetic biology labels of shape `(N,)`.
        target_x: Target feature matrix of shape `(M, d)`.
        target_labels: Target biology labels of shape `(M,)`.

    Returns:
        A mapping from integer label id to centroid Euclidean distance.
    """
    x_np = _as_numpy_2d(x, name="x")
    labels_np = _as_numpy_1d(labels, name="labels")
    target_x_np = _as_numpy_2d(target_x, name="target_x")
    target_labels_np = _as_numpy_1d(target_labels, name="target_labels")
    if x_np.shape[1] != target_x_np.shape[1]:
        raise ValueError(f"Feature dim mismatch: {x_np.shape[1]} vs {target_x_np.shape[1]}")
    if x_np.shape[0] != labels_np.shape[0]:
        raise ValueError(f"x and labels must align, got {x_np.shape[0]} vs {labels_np.shape[0]}")
    if target_x_np.shape[0] != target_labels_np.shape[0]:
        raise ValueError(
            f"target_x and target_labels must align, got {target_x_np.shape[0]} vs {target_labels_np.shape[0]}"
        )

    out: Dict[int, float] = {}
    shared_labels = np.intersect1d(np.unique(labels_np), np.unique(target_labels_np))
    for label in shared_labels.tolist():
        x_mask = labels_np == label
        target_mask = target_labels_np == label
        if not np.any(x_mask) or not np.any(target_mask):
            continue
        centroid_x = x_np[x_mask].mean(axis=0)
        centroid_target = target_x_np[target_mask].mean(axis=0)
        out[int(label)] = float(np.linalg.norm(centroid_x - centroid_target))
    return out
