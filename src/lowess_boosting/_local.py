"""Numerical core: weighted local linear fits evaluated at query points.

Everything about distances, neighbourhoods, kernels and the per-point weighted
least-squares solve lives behind :func:`local_linear_predict`. The estimators
only decide *what* to fit; this module decides *how*.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _tricubic(u: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - u**3, 0.0, None) ** 3


def _epanechnikov(u: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - u**2, 0.0, None)


def _quartic(u: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - u**2, 0.0, None) ** 2


def _gaussian(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u**2)


# Normalising constants are dropped: they cancel in weighted least squares.
KERNELS = {
    "tricubic": _tricubic,
    "epanechnikov": _epanechnikov,
    "quartic": _quartic,
    "gaussian": _gaussian,
}


# Upper bound on the number of entries in one (queries x training points) block.
_BLOCK_ENTRIES = 2**21


def local_linear_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    sample_weight: np.ndarray,
    *,
    kernel: str,
    ridge: float,
    n_neighbors: int | None = None,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Evaluate a kernel-weighted local linear fit at each query point.

    Exactly one of ``n_neighbors`` and ``bandwidth`` must be given.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
    y_train : ndarray of shape (n_samples,)
    X_query : ndarray of shape (n_queries, n_features)
    sample_weight : ndarray of shape (n_samples,)
        Non-negative per-training-point weights multiplied into the kernel
        weights (the robustness weights in LOWESS).
    kernel : str
        Key of :data:`KERNELS`.
    ridge : float
        Penalty on the local slopes in locally standardised units (never on
        the local intercept).
    n_neighbors : int, optional
        Each query's bandwidth is the distance to its ``n_neighbors``-th
        nearest training point.
    bandwidth : float, optional
        Fixed bandwidth shared by all queries.

    Returns
    -------
    ndarray of shape (n_queries,)
        Local intercepts, i.e. the fitted values at the query points. Queries
        whose window holds no positively weighted training point get the mean
        target of the nearest training points with positive ``sample_weight``.
    """
    # Centre on the training mean and divide by the largest absolute
    # coordinate. Every result below is invariant to this change of units, and
    # it keeps the squared terms finite for inputs as large as ~1e154 and the
    # uncentred moment sums well conditioned.
    offset = X_train.mean(axis=0)
    X_train = X_train - offset
    X_query = X_query - offset
    unit = float(np.max(np.abs(X_train)))
    if unit > 0:
        X_train = X_train / unit
        X_query = X_query / unit
        if bandwidth is not None:
            bandwidth = bandwidth / unit

    n_samples, n_features = X_train.shape
    Z = np.hstack([np.ones((n_samples, 1)), X_train])
    q = n_features + 1
    # Row i holds the flattened outer product z_i z_i^T, so that the weighted
    # normal equations of every query in a block come from two matrix products.
    ZZ = (Z[:, :, None] * Z[:, None, :]).reshape(n_samples, q * q)
    Zy = Z * y_train[:, None]

    out = np.empty(X_query.shape[0])
    block = max(1, _BLOCK_ENTRIES // max(n_samples, q * q))
    for start in range(0, X_query.shape[0], block):
        Xq = X_query[start : start + block]
        dist = cdist(Xq, X_train)
        W = _kernel_weights(dist, kernel, n_neighbors, bandwidth) * sample_weight

        empty = W.sum(axis=1) <= 0.0
        # Some BLAS builds (Accelerate) raise spurious floating-point flags in
        # matmul on finite inputs.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            A = (W @ ZZ).reshape(-1, q, q)
            b = W @ Zy
        A[empty] = np.eye(q)
        b[empty] = 0.0
        pred = _solve_intercepts(A, b, Xq, ridge)
        if empty.any():
            pred[empty] = _nearest_trusted_mean(dist[empty], y_train, sample_weight)
        out[start : start + block] = pred
    return out


def _kernel_weights(
    dist: np.ndarray, kernel: str, n_neighbors: int | None, bandwidth: float | None
) -> np.ndarray:
    if bandwidth is not None:
        u = dist / bandwidth
        # The compact kernels vanish at u = 1. With a fixed width the gaussian
        # is cut off at u = 4; in nearest-neighbour mode every kernel is cut
        # off at the neighbourhood edge (u = 1).
        cutoff = 4.0 if kernel == "gaussian" else 1.0
    else:
        assert n_neighbors is not None
        k = n_neighbors - 1
        h = np.partition(dist, k, axis=1)[:, k : k + 1]
        # h == 0 when at least n_neighbors training points coincide with the
        # query: those points get full weight and everything else none.
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(
                h > 0, dist / np.where(h > 0, h, 1.0), np.where(dist > 0, np.inf, 0.0)
            )
        cutoff = 1.0
    return np.where(u <= cutoff, KERNELS[kernel](np.minimum(u, cutoff)), 0.0)


def _solve_intercepts(
    A: np.ndarray, b: np.ndarray, Xq: np.ndarray, ridge: float
) -> np.ndarray:
    """Fitted value at each query from its weighted normal equations.

    ``A`` and ``b`` are the uncentred moments ``sum w z z^T`` and
    ``sum w z y`` with ``z = [1, x]``. The intercept is profiled out, the
    slopes are solved in locally standardised coordinates with the penalty
    ``ridge * I``, and directions the neighbourhood does not span (eigenvalues
    at round-off level) get zero slope. That is the same as penalising each
    raw slope by ``ridge`` times its feature's weighted scatter, so the result
    does not depend on the units of any feature.
    """
    s0 = A[:, 0, 0]
    mean_x = A[:, 0, 1:] / s0[:, None]
    mean_y = b[:, 0] / s0
    second = A[:, 1:, 1:]
    scatter = second - s0[:, None, None] * (mean_x[:, :, None] * mean_x[:, None, :])
    cross = b[:, 1:] - s0[:, None] * mean_x * mean_y[:, None]

    # A feature whose local scatter is at round-off level relative to its
    # uncentred second moment does not vary in this window.
    var = np.einsum("ijj->ij", scatter)
    varies = var > 1e-12 * np.einsum("ijj->ij", second)
    scale = np.where(varies, np.sqrt(np.where(varies, var, 1.0)), np.inf)
    corr = scatter / (scale[:, :, None] * scale[:, None, :])
    corr += ridge * np.eye(corr.shape[1])

    vals, vecs = np.linalg.eigh(corr)
    keep = vals > 1e-10 * np.maximum(vals[:, -1:], 1.0)
    inv = np.where(keep, 1.0 / np.where(keep, vals, 1.0), 0.0)
    rhs = np.einsum("ikj,ik->ij", vecs, cross / scale)
    slopes = np.einsum("ijk,ik->ij", vecs, inv * rhs) / scale
    return mean_y + np.einsum("ij,ij->i", slopes, Xq - mean_x)


def _nearest_trusted_mean(
    dist: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    """Mean target of each row's nearest training points with positive weight."""
    # If no point had positive weight every distance is inf, so every point
    # ties as nearest and the result is the overall mean.
    dist = np.where(sample_weight > 0, dist, np.inf)
    # Distances within round-off of the minimum count as ties.
    nearest = dist <= dist.min(axis=1, keepdims=True) * (1 + 1e-9)
    return (nearest @ y_train) / nearest.sum(axis=1)


def bisquare_robustness_weights(residuals: np.ndarray, scale: float) -> np.ndarray:
    """Cleveland's bisquare robustness weights from fit residuals.

    Residuals larger than six median absolute residuals get weight zero.
    ``scale`` sets a floor on the median absolute residual so that an exact
    fit (median residual at round-off level) does not turn round-off into
    outliers.
    """
    s = max(float(np.median(np.abs(residuals))), 1e-12 * scale)
    if s == 0.0:
        return np.ones_like(residuals)
    return np.clip(1.0 - (residuals / (6.0 * s)) ** 2, 0.0, None) ** 2
