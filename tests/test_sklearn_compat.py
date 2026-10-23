"""scikit-learn API conformance: the official estimator checks plus usage in
pipelines and model selection."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from lowess_boosting import LowessRegressor


@parametrize_with_checks([LowessRegressor()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
