# lowess-boosting

LOWESS (locally weighted linear regression) for any number of features, and
least-squares gradient boosting that uses LOWESS as its base learner. Both are
scikit-learn estimators: they pass `sklearn.utils.estimator_checks`, and they
work with `Pipeline`, `GridSearchCV`, `clone` and pickling.

The only runtime dependencies are numpy, scipy and scikit-learn.

## Install

```bash
pip install git+https://github.com/Siyuan-Zhu/lowess-boosting-from-scratch
```

Python 3.10 or newer. To run the tests or benchmarks from a clone:
`pip install -e ".[dev,bench]"`.

## Usage

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lowess_boosting import LowessBoostingRegressor, LowessRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(500, 2))
y = np.sin(X[:, 0]) * X[:, 1] + rng.normal(scale=0.2, size=500)

for model in [
    LowessRegressor(frac=0.3),
    LowessBoostingRegressor(LowessRegressor(frac=0.3), n_estimators=5),
]:
    pipe = make_pipeline(StandardScaler(), model)
    print(type(model).__name__, cross_val_score(pipe, X, y, cv=5).mean().round(3))
```

Output (numpy 2.5.4, scikit-learn 1.9.1):

```
LowessRegressor 0.869
LowessBoostingRegressor 0.93
```

At `frac=0.3` a single LOWESS fit oversmooths this surface, and five boosting
stages recover most of the lost detail. A single LOWESS with `frac=0.1` scores
0.954 on the same data, so tune the span before adding stages.

## API

### `LowessRegressor(frac=1/3, bandwidth=None, kernel="tricubic", robust_iters=0, ridge=1e-3)`

Each prediction is the intercept of a weighted least-squares line fitted
around the query point.

| Parameter | Meaning |
|---|---|
| `frac` | Share of the training set in each neighbourhood. The bandwidth at a query is the distance to its `ceil(frac * n)`-th nearest training point. |
| `bandwidth` | Fixed kernel width in feature units. Overrides `frac` when set. |
| `kernel` | `"tricubic"`, `"epanechnikov"`, `"quartic"` or `"gaussian"`, applied to `u = distance / bandwidth`. |
| `robust_iters` | Cleveland's bisquare robustifying iterations. Points with residuals above six median absolute residuals get zero weight. |
| `ridge` | L2 penalty on the local slopes, never on the intercept. Keeps neighbourhoods with fewer than `n_features + 1` distinct points solvable. |

Fitted attributes: `X_fit_`, `y_fit_`, `n_neighbors_`, `robustness_weights_`,
`n_features_in_`.

Distances are Euclidean on the raw inputs, so put a `StandardScaler` in front
when features have different units.

### `LowessBoostingRegressor(estimator=None, n_estimators=5, learning_rate=0.5, subsample=1.0, residual_estimator=None, random_state=None)`

Starts from the mean of `y`, then adds `n_estimators` clones of `estimator`
(default `LowessRegressor()`), each fitted to the current residuals and scaled
by `learning_rate`. `subsample < 1` fits each stage on a random subset of rows.
`residual_estimator`, if given, is fitted once to what is left at the end.

`staged_predict(X)` yields the prediction after each stage. `train_score_`
holds the in-sample MSE after each stage.

The base learner can be any scikit-learn regressor. Pass it explicitly
(`LowessBoostingRegressor(LowessRegressor())`) if you want to tune its
parameters as `estimator__frac` in a grid search.

## Design notes

### Numerical core

All the numerical work sits in one private function,
`_local.local_linear_predict`. For a block of query points it computes the
distances with `scipy.spatial.distance.cdist`, finds each query's k-th
neighbour distance with `np.partition`, applies the kernel and then builds
every query's weighted normal equations with two matrix products against
precomputed per-sample outer products `z_i z_i^T`. The batched systems are
solved with `np.linalg.solve`. Blocks are capped at about 4 million
query-by-training entries, so memory stays bounded for large test sets.

### Degenerate neighbourhoods

With `ridge=0`, a neighbourhood of fewer than `n_features + 1` distinct points
makes the local system singular. The solver then profiles out the intercept
and takes minimum-norm slopes, which is the limit of the ridge solution as the
penalty goes to zero. Slopes in directions the neighbourhood does not span are
set to zero, so when all weighted points sit at one location the prediction is
their weighted mean. If a query has no positively weighted training point at
all (a narrow fixed bandwidth, or robustness weights that zero a whole
neighbourhood), it gets the mean target of the nearest training points that
still have positive weight.

### Robustifying iterations

These follow Cleveland (1979) and the structure of Alexandre Gramfort's
univariate gist: fit at every training point, compute
bisquare weights from the residuals, and multiply them into the kernel weights
for the next pass and for prediction. The median absolute residual has a small
floor relative to `max|y|`, so an exact fit does not turn round-off into
outliers.

### Changes from the course notebooks

This package began as three DATA 441 projects (see Acknowledgements). The methods are the same: k-nearest
neighbour spans, the four kernels, robustifying iterations, a small ridge term
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
- The ridge term was added to the whole normal matrix, intercept included, so
  predictions changed when a constant was added to `y`. Only slopes are
  penalised now, and a test checks that shifting `X` and `y` shifts the
  predictions by the same constant.
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

### scikit-learn checks

The test suite runs the full `parametrize_with_checks` suite on five
configurations: default LOWESS, LOWESS with robust iterations, LOWESS with a
fixed gaussian bandwidth, default boosting, and boosting with subsampling and
a residual estimator. Every check passes and none is marked as an expected
failure. scikit-learn skips its array-API input check unless
`SCIPY_ARRAY_API=1` is set; with the variable set, that check passes too.

## Benchmarks

These numbers come from one run of the script on 2026-10-23:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python benchmarks/run_benchmarks.py
```

Install the `bench` extra first. Results go to `benchmarks/results/`
(gitignored). `--plot` also saves a per-fold RMSE chart, and `--render`
rebuilds the table from the saved `results.json` without refitting.

Settings: repeated K-fold with 5 folds x 3 repeats (15 test folds), seed 0,
every model behind a `StandardScaler`, all models on the same folds. LOWESS
`frac` (from 0.02, 0.05, 0.1, 0.2, 0.4) and KNN `n_neighbors` (5, 10, 20, 40)
are picked per outer fold by a 3-fold inner grid search on the training part.
Everything else is fixed: the boosting model uses 5 stages at learning rate
0.5, and the "course setup" row is LOWESS at `frac=1/6` followed by a
100-tree, depth-7 random forest on its residuals. Single-threaded BLAS,
Python 3.13.15, numpy 2.5.4, scikit-learn 1.9.1, Apple arm64. The whole run
took 484 s.

California housing is a seeded 5,000-row sample of the 20,640 rows. A full-data
run was started first and stopped: the tuned boosting model refits LOWESS
at every training row in every stage, and after 45 minutes single-threaded it
had not finished the 15 folds for that one model.

Auto MPG uses the three predictors from the course project (cylinders,
displacement, weight). The datasets come from OpenML (ids 196 and 4353) and
scikit-learn at run time and are not stored in this repository.

### Auto MPG

n=398, p=3, target: mpg

| Model | RMSE mean ± sd | RMSE median | Worst fold | R² (mean) | Fit + predict, s/fold |
|---|---|---|---|---|---|
| LOWESS (tuned frac) | 4.082 ± 0.390 | 4.071 | 4.913 | 0.723 | 0.02 |
| LOWESS boosting, 5 stages (tuned frac) | 4.117 ± 0.398 | 4.054 | 4.909 | 0.718 | 0.11 |
| Gradient boosting (sklearn defaults) | 4.179 ± 0.408 | 4.222 | 4.988 | 0.709 | 0.02 |
| KNN (tuned k) | 4.194 ± 0.383 | 4.129 | 4.962 | 0.708 | 0.02 |
| LOWESS + random forest on residuals (course setup) | 4.208 ± 0.411 | 4.323 | 4.989 | 0.705 | 0.04 |
| HistGradientBoosting (sklearn defaults) | 4.262 ± 0.499 | 4.185 | 5.112 | 0.698 | 0.02 |
| Linear regression | 4.316 ± 0.376 | 4.175 | 5.045 | 0.691 | 0.00 |
| Random forest (300 trees) | 4.360 ± 0.442 | 4.416 | 5.009 | 0.683 | 0.11 |

### Concrete strength

n=1030, p=8, target: MPa

| Model | RMSE mean ± sd | RMSE median | Worst fold | R² (mean) | Fit + predict, s/fold |
|---|---|---|---|---|---|
| HistGradientBoosting (sklearn defaults) | 4.365 ± 0.471 | 4.404 | 5.080 | 0.930 | 0.05 |
| Random forest (300 trees) | 4.879 ± 0.508 | 4.841 | 5.808 | 0.913 | 0.40 |
| LOWESS + random forest on residuals (course setup) | 5.016 ± 0.448 | 4.930 | 6.095 | 0.908 | 0.12 |
| Gradient boosting (sklearn defaults) | 5.065 ± 0.353 | 5.124 | 5.710 | 0.907 | 0.06 |
| LOWESS boosting, 5 stages (tuned frac) | 6.356 ± 0.335 | 6.260 | 6.925 | 0.853 | 0.71 |
| LOWESS (tuned frac) | 6.668 ± 0.359 | 6.654 | 7.422 | 0.838 | 0.06 |
| KNN (tuned k) | 8.749 ± 0.372 | 8.797 | 9.353 | 0.722 | 0.02 |
| Linear regression | 10.482 ± 0.459 | 10.513 | 11.407 | 0.602 | 0.00 |

### California housing (5000-row sample)

n=5000, p=8, target: median value, $100k

| Model | RMSE mean ± sd | RMSE median | Worst fold | R² (mean) | Fit + predict, s/fold |
|---|---|---|---|---|---|
| HistGradientBoosting (sklearn defaults) | 0.506 ± 0.025 | 0.499 | 0.541 | 0.805 | 0.07 |
| Gradient boosting (sklearn defaults) | 0.549 ± 0.025 | 0.545 | 0.583 | 0.770 | 0.42 |
| Random forest (300 trees) | 0.554 ± 0.026 | 0.548 | 0.596 | 0.766 | 2.75 |
| KNN (tuned k) | 0.666 ± 0.029 | 0.664 | 0.712 | 0.661 | 0.18 |
| LOWESS + random forest on residuals (course setup) | 0.849 ± 0.887 | 0.572 | 4.136 | -0.127 | 1.33 |
| LOWESS boosting, 5 stages (tuned frac) | 0.853 ± 0.840 | 0.609 | 3.984 | -0.073 | 24.31 |
| LOWESS (tuned frac) | 0.871 ± 0.899 | 0.601 | 4.220 | -0.172 | 1.46 |
| Linear regression | 1.369 ± 1.423 | 0.729 | 5.762 | -1.896 | 0.00 |

### Reading the results

- Auto MPG has 3 features and 398 rows. Tuned LOWESS has the lowest mean
  RMSE, but all eight models are within 0.3 mpg of each other and the
  fold-to-fold spread is about 0.4, so no model clearly wins.
- On concrete strength the tree ensembles win. Five boosting stages take
  LOWESS from 6.67 to 6.36 RMSE, still well behind HistGradientBoosting at
  4.37. The course setup reaches 5.02, about what the random forest gets on its
  own (4.88), so the forest is doing most of the work.
- On the California sample the tree ensembles win again. The three LOWESS
  models have a median fold RMSE between 0.57 and 0.61, but one fold scores
  about 4. That fold's test set contains a block group whose average occupancy
  is 1,243 (the training maximum is 41). LOWESS extends the local linear fit
  out to it and predicts 133 for a true value of 1.4. Linear regression has
  three bad folds, that one included. KNN and the tree ensembles build
  predictions from averages of training targets or residuals, so an extreme
  input cannot push them far outside the training range.
- Cost: the boosting model takes 24 s per fold on 4,000 training rows.
  HistGradientBoosting takes 0.07 s.

## Limitations

- LOWESS stores the training set, and predicting costs
  `O(n_queries * n_train * n_features^2)`. With `robust_iters=0`, fitting
  only validates and stores the data. Robustifying iterations and every
  boosting stage refit at all training points, which costs `O(n_train^2)`.
- The solver precomputes an `n_train x (n_features + 1)^2` array. With 50
  features and 20,000 rows that is about 400 MB.
- Neighbourhoods are Euclidean balls with no feature weighting, so every
  feature counts equally in the distance and irrelevant features dilute the
  neighbourhood. As the number of features grows, a fixed share of the data
  covers a wider region and the fit gets less local. Scale the inputs, and
  drop features that do not matter.
- No `sample_weight`, sparse input, missing values (NaN is rejected),
  multi-output targets or classification.
- Local linear fits extrapolate linearly. A query far outside the training
  data gets its neighbourhood's slope extended out to it, which is how one
  extreme California row produced a prediction 100 times too large (see
  Benchmarks). Clip or transform heavy-tailed features first.
- On the two larger benchmark datasets, scikit-learn's tree ensembles are
  more accurate and much faster. LOWESS was competitive only on Auto MPG, the
  smallest dataset and the only one with 3 features.
- `LowessBoostingRegressor` supports squared error only and has no early
  stopping. Use `staged_predict` on a validation set to pick `n_estimators`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy
pytest --cov
```

CI runs ruff, mypy and pytest on Python 3.10 to 3.13, then builds the wheel
and imports it from a clean virtual environment.

## Acknowledgements

This package started as coursework for DATA 441 (Advanced Applied Machine
Learning) at William & Mary. The robustifying iterations follow the structure
of Alexandre Gramfort's
[univariate LOWESS gist](https://gist.github.com/agramfort/850437) (BSD
3-clause). See [NOTICE](NOTICE) for the license text.

References:

- W. S. Cleveland (1979). Robust locally weighted regression and smoothing
  scatterplots. *JASA* 74(368), 829-836.
- W. S. Cleveland and S. J. Devlin (1988). Locally weighted regression: an
  approach to regression analysis by local fitting. *JASA* 83(403), 596-610.
- J. H. Friedman (2001). Greedy function approximation: a gradient boosting
  machine. *Annals of Statistics* 29(5), 1189-1232.

## License

MIT. See [LICENSE](LICENSE).
