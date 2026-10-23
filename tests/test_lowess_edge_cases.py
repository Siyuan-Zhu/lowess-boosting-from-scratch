"""Edge cases: tiny or degenerate training sets and extreme smoothing settings."""

import numpy as np
import pytest

from lowess_boosting import LowessRegressor


def test_single_training_point_predicts_its_target_everywhere():
    model = LowessRegressor().fit([[1.0, 2.0]], [5.0])

    np.testing.assert_allclose(model.predict([[1.0, 2.0], [-3.0, 8.0]]), [5.0, 5.0])


def test_two_points_with_frac_one_use_only_the_nearer_point():
    # k = 2: the farther point sits on the bandwidth with weight 0, so only the
    # nearer point counts. With no ridge the slope is unidentified and the fit
    # falls back to a local constant.
    model = LowessRegressor(frac=1.0, ridge=0.0).fit([[0.0], [2.0]], [1.0, 3.0])

    np.testing.assert_allclose(model.predict([[0.4], [1.6]]), [1.0, 3.0])


def test_duplicate_inputs_average_their_targets():
    X = np.array([[0.0], [1.0], [1.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 4.0, 5.0, 6.0, 2.0, 3.0])

    # frac=0.5 -> k=3: the three copies of x=1 are the whole window at x=1.
    pred = LowessRegressor(frac=0.5).fit(X, y).predict([[1.0]])

    np.testing.assert_allclose(pred, [5.0])


def test_all_inputs_identical_gives_the_mean_target():
    X = np.zeros((8, 2))
    y = np.arange(8.0)

    pred = LowessRegressor(ridge=0.0).fit(X, y).predict([[0.0, 0.0], [1.0, -1.0]])

    np.testing.assert_allclose(pred, [3.5, 3.5])


@pytest.mark.parametrize("robust_iters", [0, 3])
def test_constant_target_is_predicted_exactly(robust_iters):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = np.full(30, -2.5)

    model = LowessRegressor(robust_iters=robust_iters).fit(X, y)

    np.testing.assert_allclose(model.predict(rng.normal(size=(5, 2))), -2.5)
    np.testing.assert_allclose(model.robustness_weights_, 1.0, atol=1e-6)


def test_all_zero_target_with_robust_iterations():
    model = LowessRegressor(robust_iters=2).fit(
        np.arange(6.0).reshape(-1, 1), np.zeros(6)
    )

    np.testing.assert_array_equal(model.predict([[2.5]]), [0.0])


def test_huge_bandwidth_with_epanechnikov_approaches_ordinary_least_squares():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(40, 2))
    y = X @ [1.0, -2.0] + 0.3 + 0.5 * np.sin(5 * X[:, 0])
    X_new = rng.uniform(-1, 1, size=(5, 2))
    design = np.c_[np.ones(40), X]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]

    model = LowessRegressor(bandwidth=1e6, kernel="epanechnikov", ridge=0.0)
    pred = model.fit(X, y).predict(X_new)

    np.testing.assert_allclose(pred, np.c_[np.ones(5), X_new] @ beta, rtol=1e-6)


def test_tiny_frac_interpolates_the_training_targets():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 3))
    y = rng.normal(size=25)

    # k = 1: each training point is its own neighbourhood.
    pred = LowessRegressor(frac=1e-6).fit(X, y).predict(X)

    np.testing.assert_allclose(pred, y)


def test_large_query_sets_match_predicting_in_small_batches():
    # 20,000 training rows make the solver split 450 queries into several
    # internal blocks; the result must not depend on how queries are grouped.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20_000, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    X_new = rng.normal(size=(450, 2))
    model = LowessRegressor(frac=0.01).fit(X, y)

    batched = np.concatenate(
        [model.predict(X_new[i : i + 50]) for i in range(0, 450, 50)]
    )

    np.testing.assert_allclose(model.predict(X_new), batched, rtol=1e-12)


@pytest.mark.parametrize(("frac", "n", "expected"), [(0.07, 100, 7), (0.14, 50, 7)])
def test_span_size_is_not_inflated_by_floating_point_error(frac, n, expected):
    # 0.07 * 100 evaluates to 7.000000000000001 in floating point.
    X = np.arange(float(n)).reshape(-1, 1)

    model = LowessRegressor(frac=frac).fit(X, np.zeros(n))

    assert model.n_neighbors_ == expected


def test_predictions_are_invariant_to_translating_inputs_and_targets():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 2))
    y = np.cos(X[:, 0]) * X[:, 1]
    X_new = rng.normal(size=(6, 2))
    shift = np.array([1e3, -5e2])

    base = LowessRegressor(frac=0.4).fit(X, y).predict(X_new)
    moved = LowessRegressor(frac=0.4).fit(X + shift, y + 10.0).predict(X_new + shift)

    np.testing.assert_allclose(moved, base + 10.0, rtol=1e-8, atol=1e-8)


def test_single_neighbour_predictions_are_training_targets():
    # Property: with a single neighbour the prediction is a training target.
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 1))
    y = rng.normal(size=20)

    pred = LowessRegressor(frac=0.01).fit(X, y).predict(rng.normal(size=(50, 1)))

    assert np.isin(pred, y).all()


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"frac": 0.0}, "frac"),
        ({"frac": 1.5}, "frac"),
        ({"frac": "a"}, "frac"),
        ({"robust_iters": -1}, "robust_iters"),
        ({"robust_iters": 1.5}, "robust_iters"),
        ({"robust_iters": True}, "robust_iters"),
        ({"ridge": -1.0}, "ridge"),
        ({"ridge": float("nan")}, "ridge"),
        ({"frac": True}, "frac"),
        ({"ridge": False}, "ridge"),
        ({"bandwidth": True}, "bandwidth"),
    ],
)
def test_invalid_parameters_are_rejected_at_fit(params, match):
    with pytest.raises(ValueError, match=match):
        LowessRegressor(**params).fit([[0.0], [1.0]], [0.0, 1.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_inputs_are_rejected(bad):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = LowessRegressor().fit(X, y)

    with pytest.raises(ValueError):
        LowessRegressor().fit(np.where(X == 1.0, bad, X), y)
    with pytest.raises(ValueError):
        LowessRegressor().fit(X, np.where(y == 1.0, bad, y))
    with pytest.raises(ValueError):
        model.predict([[bad]])


def test_wrong_number_of_features_at_predict_is_rejected():
    model = LowessRegressor().fit(np.zeros((4, 2)), np.arange(4.0))

    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((1, 3)))
