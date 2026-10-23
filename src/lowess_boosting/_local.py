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


def local_linear_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_neighbors: int,
    kernel: str,
    ridge: float,
) -> np.ndarray:
    """Evaluate a kernel-weighted local linear fit at each query point.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
    y_train : ndarray of shape (n_samples,)
    X_query : ndarray of shape (n_queries, n_features)
    sample_weight : ndarray of shape (n_samples,)
        Per-training-point weights multiplied into the kernel weights
        (the robustness weights in LOWESS).
    n_neighbors : int
        Size of the neighbourhood that sets each query's bandwidth.
    kernel : str
        Key of :data:`KERNELS`.
    ridge : float
        L2 penalty on the local slopes (never on the local intercept).

    Returns
    -------
    ndarray of shape (n_queries,)
    """
    dist = cdist(X_query, X_train)
    h = np.partition(dist, n_neighbors - 1, axis=1)[:, n_neighbors - 1]
    u = dist / h[:, None]
    weights = np.where(u <= 1.0, KERNELS[kernel](u), 0.0) * sample_weight

    n_features = X_train.shape[1]
    out = np.empty(X_query.shape[0])
    for i in range(X_query.shape[0]):
        if not weights[i].any():
            out[i] = _nearest_trusted_mean(dist[i], y_train, sample_weight)
            continue
        Z = np.hstack([np.ones((X_train.shape[0], 1)), X_train - X_query[i]])
        A = Z.T @ (weights[i, :, None] * Z)
        A[1:, 1:] += ridge * np.eye(n_features)
        b = Z.T @ (weights[i] * y_train)
        out[i] = np.linalg.solve(A, b)[0]
    return out


def _nearest_trusted_mean(dist, y_train, sample_weight):
    """Mean response of the nearest training points with positive weight."""
    d = np.where(sample_weight > 0, dist, np.inf)
    if not np.isfinite(d).any():
        d = dist
    return float(y_train[d == d.min()].mean())


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
