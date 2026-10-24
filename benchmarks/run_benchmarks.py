"""Repeated K-fold comparison of the LOWESS estimators against scikit-learn models.

Datasets are downloaded at runtime into scikit-learn's data home (set
``SCIKIT_LEARN_DATA`` to move it) and are never written to this repository.

    python benchmarks/run_benchmarks.py                    # all datasets
    python benchmarks/run_benchmarks.py --datasets cars    # one dataset
    python benchmarks/report.py                            # tables and figures

This script only fits models and writes ``results/results.json`` with per-fold
RMSE, MAE, R2, fit time and predict time. ``report.py`` turns that file into
the tables and figures, so they can be rebuilt without refitting.

Every model sits behind a ``StandardScaler`` and sees the same outer folds.
LOWESS ``frac`` and KNN ``n_neighbors`` are chosen per outer fold by an inner
3-fold grid search on the training part only; all other settings are fixed.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing, fetch_openml, get_data_home
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lowess_boosting
from lowess_boosting import LowessBoostingRegressor, LowessRegressor

RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- data


def load_cars():
    """Auto MPG (OpenML 196): cylinders, displacement, weight -> mpg.

    The same three predictors as the original course project.
    """
    data = fetch_openml(data_id=196, as_frame=True, parser="liac-arff")
    frame = data.frame
    X = frame[["cylinders", "displacement", "weight"]].astype(float).to_numpy()
    y = frame["class"].astype(float).to_numpy()
    return X, y


def load_concrete():
    """Concrete compressive strength (OpenML 4353, UCI 165): 8 mix features."""
    frame = fetch_openml(data_id=4353, as_frame=True, parser="pandas").frame
    target = next(c for c in frame.columns if c.startswith("Concrete compressive"))
    X = frame.drop(columns=target).astype(float).to_numpy()
    y = frame[target].astype(float).to_numpy()
    return X, y


def load_california(n_rows, seed):
    """California housing (scikit-learn), optionally a seeded row subsample."""
    X, y = fetch_california_housing(return_X_y=True)
    if n_rows and n_rows < len(y):
        rows = np.random.default_rng(seed).choice(len(y), size=n_rows, replace=False)
        X, y = X[rows], y[rows]
    return X, y


def _openml(data_id, target, drop=()):
    frame = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
    y = frame[target].astype(float).to_numpy()
    X = frame.drop(columns=[target, *drop]).astype(float).to_numpy()
    return X, y


def load_airfoil():
    """Airfoil self-noise (OpenML 43919, UCI 291): 5 features -> sound pressure, dB."""
    return _openml(43919, "pressure")


def load_energy():
    """Energy efficiency (OpenML 43918, UCI 242): 8 features -> heating load."""
    return _openml(43918, "Y1")


def load_yacht():
    """Yacht hydrodynamics (OpenML 42370, UCI 243): 6 features -> resistance."""
    return _openml(42370, "Residuary.resistance")


def load_ccpp(n_rows, seed):
    """Combined cycle power plant (UCI 294): ambient conditions -> net output, MW.

    OpenML has no copy, so the UCI zip is fetched once into scikit-learn's data
    home and the first worksheet of the xlsx file is read with the standard
    library. Optionally a seeded row subsample.
    """
    cache = Path(get_data_home()) / "uci_ccpp.zip"
    if not cache.exists():
        url = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
        with urllib.request.urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    with zipfile.ZipFile(cache) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read("CCPP/Folds5x2_pp.xlsx")))
    sheet = inner.read("xl/worksheets/sheet1.xml").decode()
    rows = [
        [float(v) for v in re.findall(r"<v>([^<]*)</v>", row)]
        for row in re.findall(r"<row [^>]*>(.*?)</row>", sheet)[1:]
    ]
    data = np.asarray(rows)
    if data.shape[1] != 5 or len(data) != 9568:
        raise RuntimeError(f"unexpected CCPP shape {data.shape}")
    if n_rows and n_rows < len(data):
        keep = np.random.default_rng(seed).choice(len(data), size=n_rows, replace=False)
        data = data[keep]
    return data[:, :4], data[:, 4]


# ------------------------------------------------------------------------- models


def build_models(seed):
    inner = KFold(3, shuffle=True, random_state=seed)

    def scaled(model):
        return make_pipeline(StandardScaler(), model)

    def tuned(model, grid):
        return GridSearchCV(scaled(model), grid, cv=inner, n_jobs=1)

    fracs = [0.02, 0.05, 0.1, 0.2, 0.4]
    return {
        "LOWESS (tuned frac)": tuned(
            LowessRegressor(), {"lowessregressor__frac": fracs}
        ),
        "LOWESS boosting, 5 stages (tuned frac)": tuned(
            LowessBoostingRegressor(
                LowessRegressor(), n_estimators=5, learning_rate=0.5
            ),
            {"lowessboostingregressor__estimator__frac": fracs},
        ),
        "LOWESS + random forest on residuals (course setup)": scaled(
            LowessBoostingRegressor(
                LowessRegressor(frac=1 / 6),
                n_estimators=1,
                learning_rate=1.0,
                residual_estimator=RandomForestRegressor(
                    n_estimators=100, max_depth=7, random_state=seed
                ),
            )
        ),
        "KNN (tuned k)": tuned(
            KNeighborsRegressor(),
            {"kneighborsregressor__n_neighbors": [5, 10, 20, 40]},
        ),
        "Random forest (300 trees)": scaled(
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
        ),
        "Gradient boosting (sklearn defaults)": scaled(
            GradientBoostingRegressor(random_state=seed)
        ),
        "HistGradientBoosting (sklearn defaults)": scaled(
            HistGradientBoostingRegressor(random_state=seed)
        ),
        "Linear regression": scaled(LinearRegression()),
    }


# ------------------------------------------------------------------------ running


def fits_per_fold(model):
    """Model fits behind one outer fold: grid x inner folds + refit, or 1."""
    if isinstance(model, GridSearchCV):
        return len(ParameterGrid(model.param_grid)) * model.cv.get_n_splits() + 1
    return 1


def evaluate(X, y, n_splits, n_repeats, seed):
    folds = list(
        RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(
            X
        )
    )
    results = {
        "n_train": [len(train) for train, _ in folds],
        "n_test": [len(test) for _, test in folds],
        "models": {},
    }
    for name, model in build_models(seed).items():
        metrics = {
            k: [] for k in ("rmse", "mae", "r2", "fit_seconds", "predict_seconds")
        }
        for train, test in folds:
            start = time.perf_counter()
            model.fit(X[train], y[train])
            fitted = time.perf_counter()
            pred = model.predict(X[test])
            done = time.perf_counter()
            resid = y[test] - pred
            metrics["rmse"].append(float(np.sqrt(np.mean(resid**2))))
            metrics["mae"].append(float(np.mean(np.abs(resid))))
            metrics["r2"].append(
                float(1 - np.sum(resid**2) / np.sum((y[test] - y[test].mean()) ** 2))
            )
            metrics["fit_seconds"].append(fitted - start)
            metrics["predict_seconds"].append(done - fitted)
        results["models"][name] = {**metrics, "fits_per_fold": fits_per_fold(model)}
        print(
            f"  {name:<52} RMSE {np.mean(metrics['rmse']):.3f}"
            f" +/- {np.std(metrics['rmse']):.3f}"
            f"   fit {np.mean(metrics['fit_seconds']):.2f} s",
            flush=True,
        )
    return results


DATASETS = ["cars", "concrete", "airfoil", "energy", "yacht", "ccpp", "california"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--california-rows",
        type=int,
        default=5000,
        help="Rows sampled from California housing (0 = all 20640).",
    )
    parser.add_argument(
        "--ccpp-rows",
        type=int,
        default=5000,
        help="Rows sampled from the combined cycle power plant data (0 = all 9568).",
    )
    args = parser.parse_args()

    # key: (title, target units, loader)
    loaders = {
        "cars": ("Auto MPG", "mpg", load_cars),
        "concrete": ("Concrete strength", "MPa", load_concrete),
        "airfoil": ("Airfoil self-noise", "dB", load_airfoil),
        "energy": ("Energy efficiency (heating)", "load units", load_energy),
        "yacht": ("Yacht hydrodynamics", "dimensionless", load_yacht),
        "ccpp": (
            "Power plant output"
            + (f" ({args.ccpp_rows}-row sample)" if args.ccpp_rows else ""),
            "MW",
            lambda: load_ccpp(args.ccpp_rows, args.seed),
        ),
        "california": (
            "California housing"
            + (f" ({args.california_rows}-row sample)" if args.california_rows else ""),
            "$100k",
            lambda: load_california(args.california_rows, args.seed),
        ),
    }
    started = time.perf_counter()
    all_results = {}
    for key in args.datasets:
        title, units, load = loaders[key]
        X, y = load()
        print(f"{title}: n={X.shape[0]}, p={X.shape[1]}", flush=True)
        all_results[title] = {
            "key": key,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "units": units,
            **evaluate(X, y, args.n_splits, args.n_repeats, args.seed),
        }

    settings = {
        "run_date": time.strftime("%Y-%m-%d"),
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "california_rows": args.california_rows,
        "ccpp_rows": args.ccpp_rows,
        "total_seconds": round(time.perf_counter() - started, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lowess_boosting": lowess_boosting.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "results.json"
    out.write_text(json.dumps({"settings": settings, "results": all_results}, indent=1))
    print("\nSettings:", json.dumps(settings))
    print(f"Saved {out}; run benchmarks/report.py for tables and figures.")


if __name__ == "__main__":
    main()
