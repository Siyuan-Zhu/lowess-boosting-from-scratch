"""Statistics for comparing models over repeated K-fold cross-validation.

Pure numpy/scipy so the helpers can be unit-tested without fitting anything.

Fold scores from repeated K-fold overlap: any two training sets share most of
their rows, so the scores are positively correlated and the naive standard
error of their mean is too small. Nadeau and Bengio (2003) correct the
variance of the mean over J folds to ``(1/J + n_test/n_train) * s^2``, and
compare against a t distribution with J - 1 degrees of freedom.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_se(scores, test_train_ratio):
    """Nadeau-Bengio standard error of the mean of ``scores``."""
    scores = np.asarray(scores, dtype=float)
    j = scores.size
    return float(np.sqrt((1.0 / j + test_train_ratio) * scores.var(ddof=1)))


def corrected_mean_ci(scores, test_train_ratio, level=0.95):
    """Mean of fold scores with a corrected two-sided CI: (mean, low, high)."""
    scores = np.asarray(scores, dtype=float)
    mean = float(scores.mean())
    half = stats.t.ppf(0.5 + level / 2, scores.size - 1) * corrected_se(
        scores, test_train_ratio
    )
    return mean, mean - float(half), mean + float(half)


def corrected_paired_test(a, b, test_train_ratio, level=0.95):
    """Corrected resampled t-test on paired fold scores ``a - b``.

    Returns a dict with the mean difference, its corrected CI, t and the
    two-sided p-value. Folds must be paired (same splits for both models).
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, low, high = corrected_mean_ci(diff, test_train_ratio, level)
    se = corrected_se(diff, test_train_ratio)
    if se == 0.0:
        t_stat = 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
    else:
        t_stat = mean / se
    p = float(2 * stats.t.sf(abs(t_stat), diff.size - 1))
    return {"diff": mean, "low": low, "high": high, "t": float(t_stat), "p": p}


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def average_ranks(scores):
    """Per-row ranks (1 = lowest score, ties share the mean rank) and column means.

    ``scores`` has one row per dataset and one column per model; lower is better.
    """
    ranks = np.vstack([stats.rankdata(row) for row in np.asarray(scores, float)])
    return ranks, ranks.mean(axis=0)


def friedman_test(scores):
    """Friedman chi-square and the Iman-Davenport F statistic, with p-values."""
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    chi2, p_chi2 = stats.friedmanchisquare(*scores.T)
    spare = n * (k - 1) - chi2  # zero when every dataset ranks the models alike
    f_stat = (n - 1) * chi2 / spare if spare > 0 else np.inf
    p_f = stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1))
    return {
        "chi2": float(chi2),
        "p_chi2": float(p_chi2),
        "f": float(f_stat),
        "p_f": float(p_f),
        "n_datasets": n,
        "n_models": k,
    }


def nemenyi_critical_difference(n_models, n_datasets, alpha=0.05):
    """Nemenyi critical difference for average ranks (Demsar 2006)."""
    q = stats.studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))
