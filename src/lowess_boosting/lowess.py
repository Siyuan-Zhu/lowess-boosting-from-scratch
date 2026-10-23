"""LOWESS regression as a scikit-learn estimator."""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ._local import local_linear_predict


class LowessRegressor(RegressorMixin, BaseEstimator):
    """Locally weighted linear regression (LOWESS) for one or more features."""

    def __init__(self, frac=1 / 3, ridge=1e-3):
        self.frac = frac
        self.ridge = ridge

    def fit(self, X, y):
        self.X_fit_ = np.asarray(X, dtype=float)
        self.y_fit_ = np.asarray(y, dtype=float)
        self.n_neighbors_ = math.ceil(self.frac * self.X_fit_.shape[0])
        return self

    def predict(self, X):
        return local_linear_predict(
            self.X_fit_,
            self.y_fit_,
            np.asarray(X, dtype=float),
            n_neighbors=self.n_neighbors_,
            ridge=self.ridge,
        )
