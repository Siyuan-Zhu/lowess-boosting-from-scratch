"""Agreement with statsmodels' reference LOWESS in one dimension.

Runs only when statsmodels is installed (``pip install -e ".[statsmodels]"``).
"""

import numpy as np
import pytest

from lowess_boosting import LowessRegressor

smoothers_lowess = pytest.importorskip("statsmodels.nonparametric.smoothers_lowess")


@pytest.mark.parametrize("frac", [0.2, 0.33, 0.5])
@pytest.mark.parametrize("it", [0, 1, 3])
def test_matches_statsmodels_lowess_at_the_training_points(frac, it):
    rng = np.random.default_rng(0)
    n = 80
    x = np.sort(rng.uniform(0, 10, n))
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    y[10] += 5.0  # an outlier, so the robustifying iterations matter

    expected = smoothers_lowess.lowess(
        y, x, frac=frac, it=it, delta=0.0, return_sorted=False
    )
    # statsmodels uses k = floor(frac * n) neighbours and `it` robustifying
    # iterations; the same k here is frac = k / n, and it == robust_iters.
    k = int(frac * n + 1e-10)
    model = LowessRegressor(frac=k / n, robust_iters=it, ridge=0.0)
    pred = model.fit(x[:, None], y).predict(x[:, None])

    np.testing.assert_allclose(pred, expected, rtol=1e-10, atol=1e-10)
