# History

## Changes from the DATA 441 course notebooks

This package began as three DATA 441 projects at William & Mary. The methods
are the same: k-nearest neighbour spans, the four kernels, robustifying iterations, a small ridge term
and boosting with LOWESS followed by another regressor. The rewrite fixes
these problems in the notebooks:

- The multivariate version applied the kernel to raw distances instead of
  distances divided by the neighbourhood radius. On standardised data the
  tricubic kernel `(1 - d^3)^3` then goes negative for distances above 1. The
  package scales by the radius, as Cleveland and Gramfort do.
- New points were predicted by interpolating the training-point fits: linear
  interpolation in 1-D, and in n-D a PCA projection to three components,
  a Delaunay triangulation and a nearest-neighbour fallback outside the convex
  hull. The package runs the local fit at each query point directly, with no
  PCA step and no fallback. The course function also built two `n x n`
  diagonal matrices for every training point; the package builds the normal
  equations for a block of queries with two matrix products.
- The ridge term `0.001 * I` was added to the whole normal matrix, intercept
  included, so predictions changed when a constant was added to `y`, and its
  effect depended on the units of `X`. The package penalises only the slopes,
  each in proportion to its feature's local weighted scatter. Tests check that
  shifting `X` and `y` shifts the predictions by the same constant and that
  rescaling `X` leaves them unchanged.
- The first notebook's fixed-width kernel used `K((x - x0) / (2 * tau))`.
  `LowessRegressor(bandwidth=h)` is the same with `h = 2 * tau`, including the
  gaussian cut-off at 4 bandwidths.
- The first notebook fitted `LinearRegression` to `diag(w) X` and `diag(w) y`.
  That leaves the intercept column unweighted and is not weighted least
  squares. The package solves the weighted normal equations.
- The notebooks set each point's bandwidth to `sorted_distances[r]` with
  `r = ceil(frac * n)` and the point itself at index 0, which is the
  (r+1)-th smallest distance. The package uses the `ceil(frac * n)`-th
  smallest distance, so at a training point the window has one point fewer.
  At `frac=1` the farthest training point gets zero weight with the compact
  kernels.
- The notebook argument `iter` counted fitting passes. `robust_iters` counts
  robustifying passes, so `iter=3` corresponds to `robust_iters=2`.
- The boosting project fitted one LOWESS stage and then a random forest or
  XGBoost on its residuals. `LowessBoostingRegressor(LowessRegressor(frac=1/6),
  n_estimators=1, learning_rate=1.0, residual_estimator=RandomForestRegressor())`
  reproduces that setup, and `n_estimators > 1` with shrinkage generalises it.
