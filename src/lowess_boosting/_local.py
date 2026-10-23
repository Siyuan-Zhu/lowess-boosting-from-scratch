"""Numerical core: weighted local linear fits evaluated at query points.

Everything about distances, neighbourhoods, kernels and the per-point weighted
least-squares solve lives behind :func:`local_linear_predict`. The estimators
only decide *what* to fit; this module decides *how*.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _tricubic(u: np.ndarray) -> np.ndarray:
    return np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)


def local_linear_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    *,
    n_neighbors: int,
    ridge: float,
) -> np.ndarray:
    """Evaluate a kernel-weighted local linear fit at each query point.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
    y_train : ndarray of shape (n_samples,)
    X_query : ndarray of shape (n_queries, n_features)
    n_neighbors : int
        Size of the neighbourhood that sets each query's bandwidth.
    ridge : float
        L2 penalty on the local slopes (never on the local intercept).

    Returns
    -------
    ndarray of shape (n_queries,)
    """
    dist = cdist(X_query, X_train)
    h = np.partition(dist, n_neighbors - 1, axis=1)[:, n_neighbors - 1]
    weights = _tricubic(dist / h[:, None])

    n_features = X_train.shape[1]
    out = np.empty(X_query.shape[0])
    for i in range(X_query.shape[0]):
        Z = np.hstack([np.ones((X_train.shape[0], 1)), X_train - X_query[i]])
        A = Z.T @ (weights[i, :, None] * Z)
        A[1:, 1:] += ridge * np.eye(n_features)
        b = Z.T @ (weights[i] * y_train)
        out[i] = np.linalg.solve(A, b)[0]
    return out
