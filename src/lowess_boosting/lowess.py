"""LOWESS regression as a scikit-learn estimator."""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ._local import KERNELS, bisquare_robustness_weights, local_linear_predict


class LowessRegressor(RegressorMixin, BaseEstimator):
    """Locally weighted linear regression (LOWESS) for one or more features."""

    def __init__(self, frac=1 / 3, kernel="tricubic", robust_iters=0, ridge=1e-3):
        self.frac = frac
        self.kernel = kernel
        self.robust_iters = robust_iters
        self.ridge = ridge

    def fit(self, X, y):
        if self.kernel not in KERNELS:
            raise ValueError(
                f"kernel must be one of {sorted(KERNELS)}, got {self.kernel!r}."
            )
        self.X_fit_ = np.asarray(X, dtype=float)
        self.y_fit_ = np.asarray(y, dtype=float)
        self.n_neighbors_ = math.ceil(self.frac * self.X_fit_.shape[0])
        self.robustness_weights_ = np.ones(self.X_fit_.shape[0])
        scale = float(np.max(np.abs(self.y_fit_)))
        for _ in range(self.robust_iters):
            residuals = self.y_fit_ - self._local_fit(self.X_fit_)
            self.robustness_weights_ = bisquare_robustness_weights(residuals, scale)
        return self

    def predict(self, X):
        return self._local_fit(np.asarray(X, dtype=float))

    def _local_fit(self, X):
        return local_linear_predict(
            self.X_fit_,
            self.y_fit_,
            X,
            self.robustness_weights_,
            n_neighbors=self.n_neighbors_,
            kernel=self.kernel,
            ridge=self.ridge,
        )
