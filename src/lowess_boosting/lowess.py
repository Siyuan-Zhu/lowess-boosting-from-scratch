"""LOWESS regression as a scikit-learn estimator."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._local import KERNELS, bisquare_robustness_weights, local_linear_predict
from ._validation import (
    as_numeric_target,
    check_choice,
    check_fraction,
    check_int,
    check_non_negative,
    check_positive_or_none,
    count_of,
)


class LowessRegressor(RegressorMixin, BaseEstimator):
    """Locally weighted linear regression (LOWESS) for one or more features.

    Each prediction is the intercept of a weighted least-squares line (or
    hyperplane) fitted around the query point. Training points are weighted by
    a kernel of their Euclidean distance to the query, scaled by the distance to
    the ``ceil(frac * n_samples)``-th nearest training point.

    Distances are Euclidean on the raw features, so put a ``StandardScaler`` in
    front of this estimator when features have different units.

    Parameters
    ----------
    frac : float, default=1/3
        Fraction of the training set in each local neighbourhood, in (0, 1].
        Larger values give smoother fits. Ignored when ``bandwidth`` is set.
    bandwidth : float or None, default=None
        Fixed kernel width ``h`` in feature units, shared by all query points.
        When None (the default), each query uses the adaptive nearest-neighbour
        width set by ``frac``.
    kernel : {"tricubic", "epanechnikov", "quartic", "gaussian"}, \
            default="tricubic"
        Weight function of the scaled distance ``u = d / h``. The first three
        are zero for ``u >= 1``. The gaussian ``exp(-u**2 / 2)`` is cut off
        outside the neighbourhood (``u > 1``) in ``frac`` mode and beyond
        ``u > 4`` in ``bandwidth`` mode. In ``frac`` mode the gaussian still
        weighs ``exp(-1/2) = 0.61`` at the edge, so the fit can jump when a
        point enters or leaves the neighbourhood; the compact kernels fall
        smoothly to zero there.
    robust_iters : int, default=0
        Number of Cleveland (1979) robustifying iterations. Each one refits at
        the training points, computes bisquare weights from the residuals and
        multiplies them into the kernel weights, so points with residuals above
        six median absolute residuals stop influencing the fit.
    ridge : float, default=1e-3
        Relative L2 penalty on the local slopes: each slope is penalised by
        ``ridge`` times its feature's weighted scatter in the neighbourhood,
        which is ridge regression on locally standardised features. Results
        therefore do not depend on the units of ``X``. The local intercept
        (the prediction) is not penalised. Slope directions the neighbourhood
        does not span (fewer distinct points than ``n_features + 1``) are set
        to zero for any ``ridge``, including 0.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    n_neighbors_ : int
        Neighbourhood size ``ceil(frac * n_samples)`` (unused when
        ``bandwidth`` is set).
    robustness_weights_ : ndarray of shape (n_samples,)
        Bisquare weights from the last robustifying iteration (all ones when
        ``robust_iters=0``).
    X_fit_ : ndarray of shape (n_samples, n_features)
        Training inputs. LOWESS is a memory-based method.
    y_fit_ : ndarray of shape (n_samples,)
        Training targets.

    Notes
    -----
    Prediction costs ``O(n_queries * n_samples * n_features**2)`` time.
    Robustifying iterations add the same cost with ``n_queries = n_samples``
    at fit time. Each prediction call builds an
    ``n_samples x (n_features + 1)**2`` array: about 140 MB peak with 20,000
    training rows and 8 features, and 540 MB with 50 features.

    If a query has no positively weighted training point in its window (a
    fixed ``bandwidth`` far from the data, or robustness weights that zero out
    a whole neighbourhood), its prediction is the mean target of the nearest
    training points that still have positive robustness weight.

    References
    ----------
    W. S. Cleveland (1979). Robust locally weighted regression and smoothing
    scatterplots. JASA 74(368), 829-836.

    Examples
    --------
    >>> import numpy as np
    >>> from lowess_boosting import LowessRegressor
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> y = np.sin(2 * np.pi * X.ravel())
    >>> model = LowessRegressor(frac=0.2).fit(X, y)
    >>> float(np.round(model.predict([[0.25]])[0], 2))
    0.97
    """

    def __init__(
        self,
        frac: float = 1 / 3,
        bandwidth: float | None = None,
        kernel: Literal["tricubic", "epanechnikov", "quartic", "gaussian"] = "tricubic",
        robust_iters: int = 0,
        ridge: float = 1e-3,
    ) -> None:
        self.frac = frac
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.robust_iters = robust_iters
        self.ridge = ridge

    def fit(self, X: ArrayLike, y: ArrayLike) -> LowessRegressor:
        """Store the training data and compute robustness weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs. NaN and infinite values are rejected.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : LowessRegressor
            Fitted estimator.
        """
        self._check_params()
        X_fit, y_raw = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        y_fit = as_numeric_target(y_raw)
        self.X_fit_ = X_fit
        self.y_fit_ = y_fit
        self.n_neighbors_ = count_of(self.frac, X_fit.shape[0])
        self.robustness_weights_ = np.ones(X_fit.shape[0])
        scale = float(np.max(np.abs(y_fit)))
        for _ in range(self.robust_iters):
            residuals = y_fit - self._local_fit(X_fit)
            self.robustness_weights_ = bisquare_robustness_weights(residuals, scale)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict targets with a local linear fit at each row of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            Predicted targets.
        """
        check_is_fitted(self)
        return self._local_fit(validate_data(self, X, dtype=np.float64, reset=False))

    def _local_fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return local_linear_predict(
            self.X_fit_,
            self.y_fit_,
            X,
            self.robustness_weights_,
            kernel=self.kernel,
            ridge=self.ridge,
            n_neighbors=None if self.bandwidth is not None else self.n_neighbors_,
            bandwidth=self.bandwidth,
        )

    def _check_params(self) -> None:
        check_choice("kernel", self.kernel, KERNELS)
        check_fraction("frac", self.frac)
        check_int("robust_iters", self.robust_iters, low=0)
        check_positive_or_none("bandwidth", self.bandwidth)
        check_non_negative("ridge", self.ridge)
