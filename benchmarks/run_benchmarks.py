"""Repeated K-fold comparison of the LOWESS estimators against scikit-learn models.

Datasets are downloaded at runtime into scikit-learn's data home (set
``SCIKIT_LEARN_DATA`` to move it) and are never written to this repository.

    python benchmarks/run_benchmarks.py                    # all datasets
    python benchmarks/run_benchmarks.py --datasets cars    # one dataset
    python benchmarks/run_benchmarks.py --plot             # also save a PNG

Every model sits behind a ``StandardScaler`` and sees the same outer folds.
LOWESS ``frac`` and KNN ``n_neighbors`` are chosen per outer fold by an inner
3-fold grid search on the training part only; all other settings are fixed.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing, fetch_openml
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
    results = {}
    for name, model in build_models(seed).items():
        rmse, r2, seconds = [], [], []
        for train, test in folds:
            start = time.perf_counter()
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            seconds.append(time.perf_counter() - start)
            resid = y[test] - pred
            rmse.append(float(np.sqrt(np.mean(resid**2))))
            r2.append(
                float(1 - np.sum(resid**2) / np.sum((y[test] - y[test].mean()) ** 2))
            )
        results[name] = {
            "rmse": rmse,
            "r2": r2,
            "seconds": seconds,
            "fits_per_fold": fits_per_fold(model),
        }
        print(
            f"  {name:<52} RMSE {np.mean(rmse):.3f} +/- {np.std(rmse):.3f}"
            f"   R2 {np.mean(r2):.3f}   {np.mean(seconds):.2f} s/fold",
            flush=True,
        )
    return results


def markdown_table(all_results):
    lines = []
    for dataset, info in all_results.items():
        lines.append(
            f"\n### {dataset}\n\nn={info['n_samples']}, p={info['n_features']}, "
            f"target: {info['target']}\n"
        )
        lines.append(
            "| Model | RMSE mean ± sd | RMSE median | Worst fold | R² (mean) "
            "| Fits per fold | Seconds per fold (all fits) "
            "| Seconds per fit (approx.) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        ranked = sorted(info["models"].items(), key=lambda kv: np.mean(kv[1]["rmse"]))
        for name, res in ranked:
            rmse = np.asarray(res["rmse"])
            lines.append(
                f"| {name} | {rmse.mean():.3f} ± {rmse.std():.3f} "
                f"| {np.median(rmse):.3f} | {rmse.max():.3f} "
                f"| {np.mean(res['r2']):.3f} | {res['fits_per_fold']} "
                f"| {np.mean(res['seconds']):.2f} "
                f"| {np.mean(res['seconds']) / res['fits_per_fold']:.2f} |"
            )
    return "\n".join(lines)


def plot(all_results, path):
    """One panel per dataset: each model's fold RMSEs relative to the best mean."""
    import matplotlib.pyplot as plt

    ink, muted, mark, surface = "#0b0b0b", "#898781", "#2a78d6", "#fcfcfb"
    names = list(next(iter(all_results.values()))["models"])
    fig, axes = plt.subplots(
        1,
        len(all_results),
        figsize=(4.2 * len(all_results), 0.45 * len(names) + 1.4),
        sharey=True,
        facecolor=surface,
    )
    axes = np.atleast_1d(axes)
    for ax, (dataset, info) in zip(axes, all_results.items(), strict=True):
        best = min(np.mean(r["rmse"]) for r in info["models"].values())
        for row, name in enumerate(names):
            rel = np.asarray(info["models"][name]["rmse"]) / best
            ax.scatter(rel, np.full(rel.size, row), s=10, color=mark, alpha=0.35, lw=0)
            ax.scatter([rel.mean()], [row], s=48, color=mark, edgecolor=surface, lw=2)
        ax.axvline(1.0, color=muted, lw=1, ls=":")
        ax.set_title(dataset, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("RMSE / best model's mean RMSE", color=muted, fontsize=9)
        ax.set_facecolor(surface)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(muted)
    axes[0].set_yticks(range(len(names)), names, color=ink, fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cars", "concrete", "california"],
        choices=["cars", "concrete", "california"],
    )
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
        "--plot", action="store_true", help="Save a PNG (needs matplotlib)."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Rebuild results.md (and the PNG with --plot) from results.json "
        "without refitting.",
    )
    args = parser.parse_args()

    if args.render:
        saved = json.loads((RESULTS_DIR / "results.json").read_text())
        table = markdown_table(saved["results"])
        (RESULTS_DIR / "results.md").write_text(table + "\n")
        print(table)
        print("\nSettings:", json.dumps(saved["settings"]))
        if args.plot:
            plot(saved["results"], RESULTS_DIR / "rmse.png")
        return

    loaders = {
        "cars": ("Auto MPG", "mpg", load_cars),
        "concrete": ("Concrete strength", "MPa", load_concrete),
        "california": (
            "California housing",
            "median value, $100k",
            lambda: load_california(args.california_rows, args.seed),
        ),
    }
    started = time.perf_counter()
    all_results = {}
    for key in args.datasets:
        title, target, load = loaders[key]
        X, y = load()
        if (
            key == "california"
            and args.california_rows
            and args.california_rows < 20640
        ):
            title += f" ({args.california_rows}-row sample)"
        print(f"{title}: n={X.shape[0]}, p={X.shape[1]}", flush=True)
        all_results[title] = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "target": target,
            "models": evaluate(X, y, args.n_splits, args.n_repeats, args.seed),
        }

    settings = {
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "california_rows": args.california_rows,
        "total_seconds": round(time.perf_counter() - started, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lowess_boosting": lowess_boosting.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps({"settings": settings, "results": all_results}, indent=2)
    )
    table = markdown_table(all_results)
    (RESULTS_DIR / "results.md").write_text(table + "\n")
    print(table)
    print("\nSettings:", json.dumps(settings))
    if args.plot:
        plot(all_results, RESULTS_DIR / "rmse.png")
        print(f"Saved {RESULTS_DIR / 'rmse.png'}")


if __name__ == "__main__":
    main()
