"""Unit tests for the benchmark statistics helpers (benchmarks/bench_stats.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_stats as bs


def test_corrected_se_matches_hand_computation():
    # scores 1..6: sample variance 3.5, J = 6, test/train = 1/4
    # variance of the mean = (1/6 + 1/4) * 3.5 = 1.458333...
    assert bs.corrected_se([1, 2, 3, 4, 5, 6], 0.25) == pytest.approx(
        np.sqrt(1.4583333333333333), rel=1e-12
    )


def test_corrected_ci_uses_t_with_j_minus_1_df():
    mean, low, high = bs.corrected_mean_ci([1, 2, 3, 4, 5, 6], 0.25)
    half = 2.570581836 * 1.2076147  # t_{0.975, 5} * corrected SE
    assert mean == 3.5
    assert (low, high) == pytest.approx((3.5 - half, 3.5 + half), rel=1e-6)


def test_correction_widens_the_naive_interval():
    scores = np.array([4.1, 3.9, 4.4, 4.0, 4.3, 3.8, 4.2, 4.5, 4.0, 3.9])
    naive = stats.sem(scores)
    assert bs.corrected_se(scores, 0.25) > naive
    assert bs.corrected_se(scores, 0.0) == pytest.approx(naive)


def test_paired_test_matches_hand_computation():
    a, b = [3, 5, 4, 6], [1, 2, 2, 2]
    # differences 2, 3, 2, 4: mean 2.75, variance 11/12
    # corrected SE = sqrt((1/4 + 1/4) * 11/12) = 0.676993, t = 2.75 / SE = 4.06202
    res = bs.corrected_paired_test(a, b, 0.25)
    assert res["diff"] == pytest.approx(2.75)
    assert res["t"] == pytest.approx(2.75 / np.sqrt(0.5 * 11 / 12))
    naive = stats.ttest_1samp(np.subtract(a, b), 0.0)
    shrink = np.sqrt((1 / 4) / (1 / 4 + 1 / 4))
    assert res["t"] == pytest.approx(naive.statistic * shrink)
    assert res["p"] == pytest.approx(2 * stats.t.sf(res["t"], 3))
    assert res["low"] < res["diff"] < res["high"]


def test_paired_test_identical_scores_gives_p_one():
    res = bs.corrected_paired_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.25)
    assert (res["diff"], res["t"], res["p"]) == (0.0, 0.0, 1.0)


def test_paired_test_constant_shift_is_infinitely_significant():
    res = bs.corrected_paired_test([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], 0.25)
    assert res["t"] == np.inf
    assert res["p"] == 0.0


def test_holm_adjustment():
    adjusted = bs.holm_adjust([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 -> .02, .03, .06, .04
    # with the running maximum the last becomes .06
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_never_lowers_p():
    p = np.array([0.4, 0.9, 0.2])
    adjusted = bs.holm_adjust(p)
    assert adjusted.max() <= 1.0
    assert np.all(adjusted >= p)


def test_average_ranks_with_ties():
    scores = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
    ranks, mean_ranks = bs.average_ranks(scores)
    assert ranks.tolist() == [[1, 2, 3], [1.5, 1.5, 3], [3, 2, 1]]
    assert mean_ranks == pytest.approx([5.5 / 3, 5.5 / 3, 7 / 3])


def test_friedman_perfect_agreement():
    # 4 datasets, 3 models always ranked 1, 2, 3: chi2 = 12N/(k(k+1)) * (sum R^2
    # - k(k+1)^2/4) = 4 * (14 - 12) = 8 with average ranks 1, 2, 3
    scores = np.tile([1.0, 2.0, 3.0], (4, 1))
    res = bs.friedman_test(scores)
    assert res["chi2"] == pytest.approx(8.0)
    assert res["p_chi2"] == pytest.approx(stats.chi2.sf(8.0, 2))
    assert res["f"] == np.inf  # chi2 = n (k - 1): the F statistic diverges
    assert res["p_f"] == 0.0


def test_pairwise_wins_counts_ties_as_half():
    scores = np.array([[1.0, 2.0, 2.0], [3.0, 1.0, 2.0]])
    wins = bs.pairwise_wins(scores)
    # model 0 beats 1 and 2 on task 0 and loses to both on task 1
    assert wins[0, 1] == 1 and wins[1, 0] == 1
    # models 1 and 2 tie on task 0; model 1 beats 2 on task 1
    assert wins[1, 2] == 1.5 and wins[2, 1] == 0.5
    assert np.all(np.diag(wins) == 0)


def test_bradley_terry_recovers_known_strengths():
    theta = np.array([0.0, 0.7, -1.2, 2.0])
    p = 1 / (1 + np.exp(theta[None, :] - theta[:, None]))  # P(i beats j)
    wins = 1e4 * p
    np.fill_diagonal(wins, 0.0)
    fit = bs.fit_bradley_terry(wins, prior=0.0)
    assert fit == pytest.approx(theta - theta.mean(), abs=1e-6)


def test_elo_scale_and_anchor():
    # 10 wins to 1 is 10:1 odds, which is 400 Elo points
    wins = np.array([[0.0, 10.0], [1.0, 0.0]])
    elo = bs.elo_ratings(wins, anchor=1, prior=0.0)
    assert elo == pytest.approx([1400.0, 1000.0], abs=1e-6)


def test_equal_records_give_equal_elo():
    wins = np.array([[0.0, 5.0], [5.0, 0.0]])
    assert bs.elo_ratings(wins, anchor=0) == pytest.approx([1000.0, 1000.0])


def test_prior_keeps_a_winless_model_finite():
    wins = np.array([[0.0, 8.0], [0.0, 0.0]])
    assert np.all(np.isfinite(bs.elo_ratings(wins, anchor=1)))


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(7, 3, 4)) + np.array([0.0, 0.5, 1.0, 1.5])
    per_dataset = np.stack([bs.pairwise_wins(s) for s in scores])
    first = bs.bootstrap_elo(per_dataset, anchor=3, n_boot=200, seed=4)
    again = bs.bootstrap_elo(per_dataset, anchor=3, n_boot=200, seed=4)
    for a, b in zip(first, again, strict=True):
        assert np.array_equal(a, b)
    point, low, high = first
    assert np.all(low <= high)
    assert np.all(low <= point + 1e-9) and np.all(point <= high + 1e-9)
    assert point[3] == pytest.approx(1000.0) and low[3] == high[3] == 1000.0
    assert point[0] > point[1] > point[2] > point[3]


def test_bootstrap_of_identical_datasets_has_zero_width():
    one = bs.pairwise_wins(np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 2.0]]))
    point, low, high = bs.bootstrap_elo(np.stack([one] * 5), anchor=2, n_boot=50)
    assert low == pytest.approx(point) and high == pytest.approx(point)


def test_pareto_front_keeps_non_dominated_models():
    cost = [1.0, 2.0, 3.0, 4.0]
    score = [10.0, 9.0, 12.0, 11.0]
    assert bs.pareto_front(cost, score) == [0, 2]


def test_wilcoxon_against_reference_uses_log_ratios():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    worse = ref * np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    rows = bs.wilcoxon_vs_reference(np.column_stack([ref, worse]), reference=0)
    assert rows[0]["beats"] == 0
    # all 7 differences have the same sign: exact two-sided p = 2 / 2**7
    assert rows[0]["p"] == pytest.approx(2 / 128)
    assert rows[0]["p_holm"] == pytest.approx(2 / 128)  # one comparison
