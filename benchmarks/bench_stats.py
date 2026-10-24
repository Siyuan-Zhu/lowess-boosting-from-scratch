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


def pairwise_wins(scores):
    """Win matrix from a (tasks x models) score array, lower score = win.

    ``wins[i, j]`` counts the tasks where model i beats model j; a tie counts
    half a win for each side. The diagonal is zero.
    """
    scores = np.asarray(scores, dtype=float)
    less = (scores[:, :, None] < scores[:, None, :]).sum(axis=0)
    equal = (scores[:, :, None] == scores[:, None, :]).sum(axis=0)
    wins = less + 0.5 * equal
    np.fill_diagonal(wins, 0.0)
    return wins


def fit_bradley_terry(wins, prior=0.5, n_iter=2000, tol=1e-12):
    """Bradley-Terry log-strengths by maximum likelihood (MM algorithm).

    ``P(i beats j) = 1 / (1 + exp(theta_j - theta_i))``. ``prior`` adds that
    many wins to each side of every pair, which keeps the estimates finite
    when a model wins or loses every comparison; use 0 for the plain MLE.
    The result is centred to mean zero.
    """
    wins = np.asarray(wins, dtype=float) + prior
    np.fill_diagonal(wins, 0.0)
    games = wins + wins.T
    total = wins.sum(axis=1)
    strength = np.ones(len(wins))
    for _ in range(n_iter):
        denom = (games / (strength[:, None] + strength[None, :])).sum(axis=1)
        new = total / denom
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(np.log(new) - np.log(strength))) < tol:
            strength = new
            break
        strength = new
    return np.log(strength)


def elo_ratings(wins, anchor, anchor_rating=1000.0, prior=0.5):
    """Bradley-Terry fit on the Elo scale (400 points = 10:1 odds), with the
    model at index ``anchor`` fixed at ``anchor_rating``."""
    theta = fit_bradley_terry(wins, prior=prior)
    return anchor_rating + 400.0 / np.log(10.0) * (theta - theta[anchor])


def bootstrap_elo(dataset_wins, anchor, n_boot=1000, seed=0, level=0.95, prior=0.5):
    """Elo ratings with a percentile CI from resampling whole datasets.

    ``dataset_wins`` has shape (datasets, models, models), one win matrix per
    dataset. Returns (point, low, high) arrays over the models.
    """
    dataset_wins = np.asarray(dataset_wins, dtype=float)
    n = len(dataset_wins)
    point = elo_ratings(dataset_wins.sum(axis=0), anchor, prior=prior)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, dataset_wins.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        draws[b] = elo_ratings(dataset_wins[pick].sum(axis=0), anchor, prior=prior)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha], axis=0)
    return point, low, high


def pareto_front(cost, score):
    """Indices of models not beaten on both lower cost and higher score,
    ordered by increasing cost."""
    cost, score = np.asarray(cost, float), np.asarray(score, float)
    front, best = [], -np.inf
    for i in np.argsort(cost, kind="stable"):
        if score[i] > best:
            front.append(int(i))
            best = score[i]
    return front


def wilcoxon_vs_reference(values, reference):
    """Paired two-sided Wilcoxon signed-rank test of each column against column
    ``reference``, on log ratios so the datasets' scales do not matter.

    ``values`` has one row per dataset and one column per model (lower is
    better). Returns one dict per other model with the number of datasets it
    beats the reference on, the statistic, the raw p-value and the Holm p.
    """
    values = np.asarray(values, dtype=float)
    others = [j for j in range(values.shape[1]) if j != reference]
    rows = []
    for j in others:
        ratio = np.log(values[:, j] / values[:, reference])
        res = stats.wilcoxon(ratio)
        rows.append(
            {
                "model": j,
                "beats": int((ratio < 0).sum()),
                "statistic": float(res.statistic),
                "p": float(res.pvalue),
            }
        )
    holm = holm_adjust([r["p"] for r in rows])
    for r, h in zip(rows, holm, strict=True):
        r["p_holm"] = float(h)
    return rows
