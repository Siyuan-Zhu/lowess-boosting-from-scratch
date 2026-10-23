"""Least-squares gradient boosting with a LOWESS base learner."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._validation import as_numeric_target, check_fraction, check_int, count_of
from .lowess import LowessRegressor

_MAX_SEED = np.iinfo(np.int32).max


class LowessBoostingRegressor(RegressorMixin, BaseEstimator):
    """Gradient boosting for squared error with LOWESS as the base learner.

    The model starts from the mean target and adds ``n_estimators`` copies of
    ``estimator``, each fitted to the current residuals (the negative gradient
    of squared error) and scaled by ``learning_rate``. An optional
    ``residual_estimator`` is then fitted once to what is left.

    With ``n_estimators=1``, ``learning_rate=1.0`` and a random forest as
    ``residual_estimator`` the model is two-stage: LOWESS first, then a tree
    ensemble on its residuals.

    Parameters
    ----------
    estimator : regressor, default=None
        Base learner cloned for every boosting stage. ``None`` means
        ``LowessRegressor()``. Any scikit-learn regressor works. Pass it
        explicitly to tune its parameters, e.g. ``estimator__frac`` in
        ``GridSearchCV``.
    n_estimators : int, default=5
        Number of boosting stages.
    learning_rate : float, default=0.5
        Shrinkage applied to each stage, in (0, 1].
    subsample : float, default=1.0
        Fraction of training rows, drawn without replacement, used to fit each
        stage (``ceil(subsample * n_samples)`` rows). Values below 1 give
        stochastic gradient boosting and, for LOWESS, cut each stage's cost
        roughly in proportion.
    residual_estimator : regressor, default=None
        Optional regressor fitted once, without shrinkage, to the residuals left
        after the last boosting stage.
    random_state : int, RandomState instance or None, default=None
        Seeds the row subsampling. Sub-estimators that expose a
        ``random_state`` parameter left at ``None`` get a seed drawn from this
        generator, so a fixed ``random_state`` makes the whole fit repeatable.

    Attributes
    ----------
    init_ : float
        Mean training target, the stage-0 prediction.
    estimators_ : list of regressors
        Fitted boosting stages.
    residual_estimator_ : regressor or None
        Fitted ``residual_estimator``.
    train_score_ : ndarray of shape (n_estimators,)
        In-sample mean squared error after each boosting stage.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    With the default LOWESS base learner, fitting costs about
    ``n_estimators * (subsample * n_samples) * n_samples`` local fits and
    predicting costs ``n_estimators * subsample * n_samples`` per query row.

    References
    ----------
    J. H. Friedman (2001). Greedy function approximation: a gradient boosting
    machine. Annals of Statistics 29(5), 1189-1232.

    Examples
    --------
    >>> import numpy as np
    >>> from lowess_boosting import LowessBoostingRegressor, LowessRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, size=(200, 2))
    >>> y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    >>> model = LowessBoostingRegressor(LowessRegressor(frac=0.2), random_state=0)
    >>> model.fit(X, y).score(X, y) > 0.99
    True
    """

    def __init__(
        self,
        estimator: Any = None,
        n_estimators: int = 5,
        learning_rate: float = 0.5,
        subsample: float = 1.0,
        residual_estimator: Any = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.residual_estimator = residual_estimator
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> LowessBoostingRegressor:
        """Fit the boosting stages and the optional residual estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs. NaN and infinite values are rejected.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : LowessBoostingRegressor
            Fitted estimator.
        """
        self._check_params()
        X_fit, y_raw = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        y_fit = as_numeric_target(y_raw)
        rng = check_random_state(self.random_state)
        n_samples = X_fit.shape[0]
        n_rows = count_of(self.subsample, n_samples)
        base = LowessRegressor() if self.estimator is None else self.estimator

        self.init_ = float(np.mean(y_fit))
        current = np.full(n_samples, self.init_)
        self.estimators_ = []
        self.train_score_ = np.empty(self.n_estimators)
        for stage in range(self.n_estimators):
            rows = (
                rng.choice(n_samples, size=n_rows, replace=False)
                if n_rows < n_samples
                else slice(None)
            )
            learner = _seeded_clone(base, rng)
            learner.fit(X_fit[rows], (y_fit - current)[rows])
            current = current + self.learning_rate * learner.predict(X_fit)
            self.estimators_.append(learner)
            self.train_score_[stage] = np.mean((y_fit - current) ** 2)

        self.residual_estimator_ = None
        if self.residual_estimator is not None:
            self.residual_estimator_ = _seeded_clone(self.residual_estimator, rng)
            self.residual_estimator_.fit(X_fit, y_fit - current)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict targets as the sum of all stages.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            Predicted targets.
        """
        return deque(self.staged_predict(X), maxlen=1)[0]

    def staged_predict(self, X: ArrayLike) -> Iterator[NDArray[np.float64]]:
        """Yield predictions after each stage.

        Yields one array per boosting stage and, when a residual estimator was
        fitted, one more that includes it. The last array equals
        :meth:`predict`.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Yields
        ------
        y_pred : ndarray of shape (n_queries,)
            Predicted targets after the stage.
        """
        check_is_fitted(self)
        X_query = validate_data(self, X, dtype=np.float64, reset=False)
        pred = np.full(X_query.shape[0], self.init_)
        for learner in self.estimators_:
            pred = pred + self.learning_rate * learner.predict(X_query)
            yield pred
        if self.residual_estimator_ is not None:
            yield pred + self.residual_estimator_.predict(X_query)

    def _check_params(self) -> None:
        check_int("n_estimators", self.n_estimators, low=1)
        check_fraction("learning_rate", self.learning_rate)
        check_fraction("subsample", self.subsample)
        for name in ("estimator", "residual_estimator"):
            value = getattr(self, name)
            if value is not None and not (
                hasattr(value, "fit") and hasattr(value, "predict")
            ):
                raise TypeError(
                    f"{name} must be a regressor with fit and predict, got {value!r}."
                )


def _seeded_clone(estimator: Any, rng: np.random.RandomState) -> Any:
    """Clone ``estimator`` and seed every unset ``random_state`` from ``rng``."""
    learner = clone(estimator)
    unset = {
        key: rng.randint(_MAX_SEED)
        for key, value in learner.get_params(deep=True).items()
        if (key == "random_state" or key.endswith("__random_state")) and value is None
    }
    return learner.set_params(**unset) if unset else learner
