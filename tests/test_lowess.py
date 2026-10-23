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
