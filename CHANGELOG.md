# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-23

### Added

- `LowessRegressor`: multivariate local linear regression with tricubic,
  Epanechnikov, quartic and gaussian kernels, a nearest-neighbour span (`frac`)
  or a fixed `bandwidth`, Cleveland's bisquare robustifying iterations and a
  slope-only ridge penalty.
- `LowessBoostingRegressor`: least-squares gradient boosting with any
  regressor as the base learner (LOWESS by default), shrinkage, row
  subsampling, an optional residual estimator and `staged_predict`.
- Both estimators pass `sklearn.utils.estimator_checks.parametrize_with_checks`.
- Benchmark script comparing against scikit-learn models with repeated K-fold
  cross-validation.
