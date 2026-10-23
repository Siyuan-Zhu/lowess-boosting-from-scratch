"""Behaviour of LowessRegressor through its public interface."""

import numpy as np
import pytest

from lowess_boosting import LowessRegressor


def test_reproduces_a_straight_line_exactly():
    # A local linear fit of noiseless linear data is the line itself,
    # whatever the kernel weights are.
    X = np.linspace(0, 10, 25).reshape(-1, 1)
    y = 3.0 - 2.0 * X.ravel()
    X_new = np.array([[0.5], [4.25], [9.9]])

    pred = LowessRegressor(frac=0.3, ridge=0.0).fit(X, y).predict(X_new)

    np.testing.assert_allclose(pred, [2.0, -5.5, -16.8], atol=1e-10)


def test_symmetric_neighbourhood_gives_hand_computed_weighted_mean():
    # Query x=2 on x=0..4 with all 5 points in the span: h=2, so the tricubic
    # weights are [0, 343/512, 1, 343/512, 0]. With symmetric weights the local
    # intercept is the weighted mean of y = [0, 1, 0, 1, 0], which is 343/599.
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

    pred = LowessRegressor(frac=1.0, ridge=0.0).fit(X, y).predict([[2.0]])

    np.testing.assert_allclose(pred, [343 / 599], rtol=1e-12)


def test_line_through_the_only_two_weighted_points():
    # With k=3 the third-nearest point sits exactly on the bandwidth and gets
    # zero tricubic weight, so the local line passes through (0, 1) and (1, 3).
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 10.0, -4.0])

    pred = LowessRegressor(frac=0.75, ridge=0.0).fit(X, y).predict([[0.5], [0.25]])

    np.testing.assert_allclose(pred, [2.0, 1.5], atol=1e-12)


def test_reproduces_a_plane_in_three_dimensions():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(60, 3))
    coef = np.array([1.5, -0.5, 2.0])
    y = 0.7 + X @ coef
    X_new = rng.uniform(-1, 1, size=(10, 3))

    pred = LowessRegressor(frac=0.4, ridge=0.0).fit(X, y).predict(X_new)

    np.testing.assert_allclose(pred, 0.7 + X_new @ coef, atol=1e-9)


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        # Same symmetric setup as above; u = |x - 2| / 2 = [1, .5, 0, .5, 1].
        ("tricubic", 343 / 599),
        # Epanechnikov: K(.5)/K(0) = 0.75 and K(1) = 0 -> 1.5 / 2.5.
        ("epanechnikov", 0.6),
        # Quartic: K(.5)/K(0) = 0.5625 and K(1) = 0 -> 1.125 / 2.125.
        ("quartic", 9 / 17),
        # Gaussian keeps the boundary point: weights exp(-u^2 / 2) for u <= 1.
        (
            "gaussian",
            2 * np.exp(-0.125) / (1 + 2 * np.exp(-0.125) + 2 * np.exp(-0.5)),
        ),
    ],
)
def test_each_kernel_weights_the_neighbourhood_as_documented(kernel, expected):
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

    model = LowessRegressor(frac=1.0, kernel=kernel, ridge=0.0).fit(X, y)

    np.testing.assert_allclose(model.predict([[2.0]]), [expected], rtol=1e-12)


def test_unknown_kernel_is_rejected_at_fit():
    with pytest.raises(ValueError, match="kernel"):
        LowessRegressor(kernel="boxcar").fit([[0.0], [1.0]], [0.0, 1.0])


def _line_with_one_outlier():
    rng = np.random.default_rng(42)
    X = np.linspace(0, 20, 41).reshape(-1, 1)
    y = 2.0 * X.ravel() + rng.normal(scale=0.5, size=41)
    y[20] += 100.0  # x = 10
    return X, y


def test_robustifying_iterations_ignore_a_gross_outlier():
    X, y = _line_with_one_outlier()

    plain = LowessRegressor(frac=0.5, robust_iters=0).fit(X, y)
    robust = LowessRegressor(frac=0.5, robust_iters=2).fit(X, y)

    assert abs(plain.predict([[10.0]])[0] - 20.0) > 5.0
    assert abs(robust.predict([[10.0]])[0] - 20.0) < 1.0
    assert robust.robustness_weights_[20] == 0.0
    assert np.median(np.delete(robust.robustness_weights_, 20)) > 0.8


def test_without_robust_iterations_all_points_keep_unit_weight():
    X, y = _line_with_one_outlier()

    model = LowessRegressor(frac=0.5).fit(X, y)

    np.testing.assert_array_equal(model.robustness_weights_, np.ones(41))


def test_fixed_bandwidth_matches_the_hand_computed_tricubic_case():
    # bandwidth=2 at x=2 gives the same scaled distances as frac=1 above.
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

    model = LowessRegressor(bandwidth=2.0, ridge=0.0).fit(X, y)

    np.testing.assert_allclose(model.predict([[2.0]]), [343 / 599], rtol=1e-12)


def test_gaussian_with_fixed_bandwidth_reaches_past_one_bandwidth():
    # u = [2, 1, 0, 1, 2]; the gaussian is only cut off beyond u = 4.
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    w1, w2 = np.exp(-0.5), np.exp(-2.0)

    model = LowessRegressor(bandwidth=1.0, kernel="gaussian", ridge=0.0).fit(X, y)

    np.testing.assert_allclose(
        model.predict([[2.0]]), [2 * w1 / (1 + 2 * w1 + 2 * w2)], rtol=1e-12
    )


def test_query_with_no_training_point_in_its_window_uses_nearest_neighbours():
    X = np.array([[0.0], [1.0], [5.0]])
    y = np.array([1.0, 2.0, 9.0])

    model = LowessRegressor(bandwidth=0.1).fit(X, y)

    np.testing.assert_allclose(model.predict([[4.8], [0.5]]), [9.0, 1.5])


@pytest.mark.parametrize("bandwidth", [0.0, -1.0, "wide"])
def test_invalid_bandwidth_is_rejected(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        LowessRegressor(bandwidth=bandwidth).fit([[0.0], [1.0]], [0.0, 1.0])
