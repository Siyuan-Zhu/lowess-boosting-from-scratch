"""Behaviour of LowessBoostingRegressor through its public interface."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from lowess_boosting import LowessBoostingRegressor, LowessRegressor


def _wiggly(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.05, size=n)
    return X, y


def test_one_full_step_equals_a_single_lowess_fit():
    # Local linear fits are equivariant to adding a constant to y, so fitting
    # y - mean(y) and adding the mean back reproduces plain LOWESS.
    X, y = _wiggly()
    base = LowessRegressor(frac=0.25)

    boosted = LowessBoostingRegressor(base, n_estimators=1, learning_rate=1.0)
    boosted.fit(X, y)

    np.testing.assert_allclose(
        boosted.predict(X[:10]), base.fit(X, y).predict(X[:10]), rtol=1e-9
    )


def test_two_stage_model_from_the_course_project():
    # LOWESS on y, then a random forest on the LOWESS residuals, summed.
    X, y = _wiggly()
    lowess = LowessRegressor(frac=1 / 6)
    forest = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0)

    boosted = LowessBoostingRegressor(
        lowess, n_estimators=1, learning_rate=1.0, residual_estimator=forest
    ).fit(X, y)

    first = LowessRegressor(frac=1 / 6).fit(X, y)
    second = clone(forest).fit(X, y - first.predict(X))
    expected = first.predict(X[:10]) + second.predict(X[:10])
    np.testing.assert_allclose(boosted.predict(X[:10]), expected, rtol=1e-9)


def test_more_stages_fit_the_training_data_more_closely():
    X, y = _wiggly()
    model = LowessBoostingRegressor(
        LowessRegressor(frac=0.5), n_estimators=6, learning_rate=0.5
    ).fit(X, y)

    single_stage_mse = np.mean(
        (y - LowessRegressor(frac=0.5).fit(X, y).predict(X)) ** 2
    )
    assert model.train_score_[-1] < 0.5 * single_stage_mse
    assert np.all(np.diff(model.train_score_) < 0)


def test_staged_predict_ends_at_predict_and_matches_train_score():
    X, y = _wiggly()
    model = LowessBoostingRegressor(
        n_estimators=3, residual_estimator=LinearRegression()
    ).fit(X, y)

    stages = list(model.staged_predict(X))

    assert len(stages) == 4
    np.testing.assert_array_equal(stages[-1], model.predict(X))
    np.testing.assert_allclose(
        [np.mean((y - p) ** 2) for p in stages[:3]], model.train_score_
    )


def test_subsampling_is_repeatable_with_a_seed_and_varies_without_one():
    X, y = _wiggly()

    def fit(seed):
        return LowessBoostingRegressor(subsample=0.5, random_state=seed).fit(X, y)

    np.testing.assert_array_equal(fit(7).predict(X), fit(7).predict(X))
    assert not np.allclose(fit(7).predict(X), fit(8).predict(X))
    assert fit(7).estimators_[0].n_features_in_ == 2
    assert fit(7).estimators_[0].X_fit_.shape[0] == 60


def test_seeds_only_sub_estimators_whose_random_state_is_unset():
    X, y = _wiggly()
    pinned = RandomForestRegressor(n_estimators=5, random_state=123)

    model = LowessBoostingRegressor(
        n_estimators=1, residual_estimator=pinned, random_state=0
    ).fit(X, y)
    free = LowessBoostingRegressor(
        n_estimators=1,
        residual_estimator=RandomForestRegressor(n_estimators=5),
        random_state=0,
    ).fit(X, y)

    assert model.residual_estimator_.random_state == 123
    assert isinstance(free.residual_estimator_.random_state, int)
    assert pinned.random_state == 123  # the user's object is never mutated


def test_any_regressor_can_be_the_base_learner():
    X, y = _wiggly()

    model = LowessBoostingRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=20)

    assert model.fit(X, y).score(X, y) > 0.8


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"n_estimators": 0}, ValueError, "n_estimators"),
        ({"n_estimators": 2.0}, ValueError, "n_estimators"),
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"learning_rate": 1.5}, ValueError, "learning_rate"),
        ({"subsample": 0.0}, ValueError, "subsample"),
        ({"subsample": 2}, ValueError, "subsample"),
        ({"estimator": "lowess"}, TypeError, "estimator"),
        ({"residual_estimator": 3}, TypeError, "residual_estimator"),
    ],
)
def test_invalid_parameters_are_rejected_at_fit(params, error, match):
    with pytest.raises(error, match=match):
        LowessBoostingRegressor(**params).fit([[0.0], [1.0]], [0.0, 1.0])


def test_nested_parameters_are_tunable_with_grid_search():
    X, y = _wiggly(n=90)
    # The base learner must be passed explicitly for its parameters to exist.
    booster = LowessBoostingRegressor(LowessRegressor(), n_estimators=2)
    pipe = make_pipeline(StandardScaler(), booster)
    grid = {"lowessboostingregressor__estimator__frac": [0.2, 1.0]}

    search = GridSearchCV(pipe, grid, cv=3).fit(X, y)

    assert search.best_params_["lowessboostingregressor__estimator__frac"] == 0.2
