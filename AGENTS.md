# AGENTS.md

scikit-learn-compatible LOWESS regression and LOWESS-based gradient boosting.
Library code is in `src/lowess_boosting`, benchmarks in `benchmarks/`, tests in `tests/`.

## Gate before every commit

```bash
pip install -e ".[dev,bench]"
ruff check . && ruff format --check . && mypy
pytest --cov          # stays at 100% branch coverage; CI fails under 95%
```

## Invariants

- Both estimators pass the whole `parametrize_with_checks` suite on all five configurations in `tests/test_sklearn_compat.py`, with no expected failures.
- In one dimension, `LowessRegressor(frac=k/n, robust_iters=K, ridge=0)` equals statsmodels `lowess` to 1e-10. Run `pip install -e ".[statsmodels]"` and `pytest tests/test_statsmodels_agreement.py` after touching `_local.py`.
- Runtime dependencies are numpy, scipy and scikit-learn.
- Every behaviour change gets a test through the public API (`fit`, `predict`, `staged_predict`, fitted attributes). Expected values come from a hand calculation or a known identity, never from the code under test.
- Statistics helpers live in `benchmarks/bench_stats.py`, with hand-computed expectations in `tests/test_benchmark_stats.py`.

## Benchmark

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python benchmarks/run_benchmarks.py    # about 12 minutes, writes results.json
python benchmarks/report.py                # tables and figures from results.json
```

- The protocol stays fixed: 5-fold cross-validation repeated 3 times, seed 0, `StandardScaler`, inner 3-fold tuning of LOWESS `frac` and KNN `k`.
- Every number and figure in the README comes from the latest `results.json`. After a rerun, update the README tables, captions and prose that quote them.
- State results as measured, including where LOWESS loses.
- Plot style lives in `benchmarks/plot_style.py`: grayscale, 7 in wide, no grid.

## Commit

- Commit the figures `fig1_elo`, `fig2_time` and `figA1_rmse` as png, svg and pdf from `benchmarks/results/`.
- Leave datasets, `results.json`, `results.md` and csv files out of git; `.gitignore` already excludes them.
- End commit messages with the `Co-Authored-By` line.
