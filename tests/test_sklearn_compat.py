"""scikit-learn API conformance: the official estimator checks plus usage in
pipelines and model selection."""

import pickle

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from lowess_boosting import LowessBoostingRegressor, LowessRegressor

ESTIMATORS = [
    LowessRegressor(),
    LowessRegressor(robust_iters=2),
    LowessRegressor(bandwidth=1.0, kernel="gaussian"),
    LowessBoostingRegressor(),
    LowessBoostingRegressor(
        LowessRegressor(frac=0.5),
        subsample=0.7,
        residual_estimator=DecisionTreeRegressor(max_depth=3),
    ),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


def _wavy_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2)) * [1.0, 100.0]
    y = np.sin(2 * X[:, 0]) + 0.01 * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


def test_clone_and_params_round_trip():
    model = LowessRegressor(frac=0.2, kernel="quartic", robust_iters=1)

    copy = clone(model).set_params(frac=0.5)

    assert copy.get_params()["frac"] == 0.5
    assert copy.get_params()["kernel"] == "quartic"
    assert model.get_params()["frac"] == 0.2


def test_grid_search_over_a_scaled_pipeline_picks_a_reasonable_span():
    X, y = _wavy_data()
    pipe = make_pipeline(StandardScaler(), LowessRegressor())
    grid = {
        "lowessregressor__frac": [0.1, 0.3, 1.0],
        "lowessregressor__kernel": ["tricubic", "gaussian"],
    }

    search = GridSearchCV(pipe, grid, cv=KFold(4, shuffle=True, random_state=0))
    search.fit(X, y)

    # The target has a full sine period per unit of x0, so the global-span
    # setting cannot win.
    assert search.best_params_["lowessregressor__frac"] < 1.0
    assert search.best_score_ > 0.8


def test_pickle_round_trip_preserves_predictions():
    X, y = _wavy_data()
    model = LowessRegressor(frac=0.3, robust_iters=1).fit(X, y)

    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.predict(X[:5]), model.predict(X[:5]))
