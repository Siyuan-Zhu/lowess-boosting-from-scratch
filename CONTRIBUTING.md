# Contributing

Bug reports and pull requests are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest --cov
```

- Add a test for every behaviour change. Tests go through the public estimators
  (`fit`, `predict`, `staged_predict`, fitted attributes), not private helpers.
- Expected values in numerical tests should come from a hand calculation or a
  known identity, not from re-running the code under test.
- Keep runtime dependencies to numpy, scipy and scikit-learn.
- If you change the benchmark script, rerun it and update the README table
  with the settings printed at the end of the run.
