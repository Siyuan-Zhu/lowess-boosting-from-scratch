"""Tables and figures from ``results/results.json`` (no refitting).

    python benchmarks/report.py

Writes ``results/results.md`` plus ``fig1_rmse`` and ``fig2_rank`` as PNG
(200 dpi) and SVG. Needs matplotlib (the ``bench`` extra).

Per dataset, each model's mean RMSE gets a 95% Nadeau-Bengio corrected
interval, and each model is compared with the best model on paired fold
scores (same splits for every model). Across datasets the models get a
Bradley-Terry Elo rating (bootstrap CI over datasets), average ranks with
Friedman's test, and Holm-adjusted Wilcoxon signed-rank tests against the
top-rated model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bench_stats as bs
import numpy as np
import plot_style

RESULTS_DIR = Path(__file__).parent / "results"
ANCHOR = "Linear regression"  # fixed at 1000 Elo
N_BOOT = 1000

SHORT = {
    "LOWESS (tuned frac)": "LOWESS",
    "LOWESS boosting, 5 stages (tuned frac)": "LOWESS boosting",
    "LOWESS + random forest on residuals (course setup)": "LOWESS + RF residuals",
    "KNN (tuned k)": "KNN",
    "Random forest (300 trees)": "Random forest",
    "Gradient boosting (sklearn defaults)": "Gradient boosting",
    "HistGradientBoosting (sklearn defaults)": "HistGradientBoosting",
    "Linear regression": "Linear regression",
}

COLUMN = {
    "yacht": "Yacht",
    "cars": "Auto MPG",
    "energy": "Energy",
    "concrete": "Concrete",
    "airfoil": "Airfoil",
    "ccpp": "Power plant",
    "california": "California",
}

# --------------------------------------------------------------------- statistics


def summarize(results, settings):
    """Per-dataset intervals and paired tests, plus the cross-dataset ranks."""
    names = list(next(iter(results.values()))["models"])
    datasets = sorted(results, key=lambda d: results[d]["n_samples"])
    table = {}
    for dataset in datasets:
        info = results[dataset]
        ratio = float(np.mean(info["n_test"]) / np.mean(info["n_train"]))
        rmse = {m: np.asarray(info["models"][m]["rmse"]) for m in names}
        best = min(names, key=lambda m: rmse[m].mean())
        others = [m for m in names if m != best]
        paired = {
            m: bs.corrected_paired_test(rmse[m], rmse[best], ratio) for m in others
        }
        holm = bs.holm_adjust([paired[m]["p"] for m in others])
        for m, p_adj in zip(others, holm, strict=True):
            paired[m]["p_holm"] = float(p_adj)
        table[dataset] = {
            "ratio": ratio,
            "best": best,
            "ci": {m: bs.corrected_mean_ci(rmse[m], ratio) for m in names},
            "paired": paired,
        }
    means = np.array(
        [[np.mean(results[d]["models"][m]["rmse"]) for m in names] for d in datasets]
    )
    ranks, avg = bs.average_ranks(means)
    # one score per dataset x repeat: the mean RMSE over that repeat's folds
    per_repeat = []
    for d in datasets:
        rmse = np.array([results[d]["models"][m]["rmse"] for m in names])
        per_repeat.append(rmse.reshape(len(names), -1, settings["n_splits"]).mean(2).T)
    dataset_wins = np.stack([bs.pairwise_wins(x) for x in per_repeat])
    anchor = names.index(ANCHOR)
    elo, elo_low, elo_high = bs.bootstrap_elo(dataset_wins, anchor, N_BOOT, seed=0)
    elo_order = [names[i] for i in np.argsort(-elo, kind="stable")]
    top = names.index(elo_order[0])
    return {
        "elo": dict(zip(names, elo, strict=True)),
        "elo_ci": {m: (elo_low[i], elo_high[i]) for i, m in enumerate(names)},
        "elo_order": elo_order,
        "n_tasks": int(sum(len(x) for x in per_repeat)),
        "wilcoxon": bs.wilcoxon_vs_reference(np.exp(np.log(means)), reference=top),
        "names": names,
        "datasets": datasets,
        "table": table,
        "ranks": ranks,
        "avg_rank": dict(zip(names, avg, strict=True)),
        "order": [names[i] for i in np.argsort(avg, kind="stable")],
        "friedman": bs.friedman_test(means),
    }


# ----------------------------------------------------------------------- tables


def decimals(results_for_dataset, summary_for_dataset):
    best = min(v[0] for v in summary_for_dataset["ci"].values())
    return max(0, 2 - math.floor(math.log10(best)))


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_text(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def markdown(results, settings, s):
    cols = " | ".join(COLUMN[results[d]["key"]] for d in s["datasets"])
    out = []
    names, order = s["names"], s["order"]
    out.append("### Mean RMSE with 95% corrected CI (half-width), by dataset\n")
    head = "| Model | " + cols + " | Average rank |"
    out += [head, "|---" * (len(s["datasets"]) + 2) + "|"]
    for m in order:
        cells = []
        for d in s["datasets"]:
            t = s["table"][d]
            dec = decimals(results[d], t)
            mean, lo, hi = t["ci"][m]
            cell = f"{mean:.{dec}f} ± {(hi - lo) / 2:.{dec}f}"
            cells.append(f"**{cell}**" if m == t["best"] else cell)
        out.append(
            f"| {SHORT[m]} | " + " | ".join(cells) + f" | {s['avg_rank'][m]:.2f} |"
        )
    out.append("\n### Average rank and Friedman test\n")
    f = s["friedman"]
    out.append(
        f"Friedman chi-square = {f['chi2']:.2f} (df {f['n_models'] - 1}, "
        f"{p_text(f['p_chi2'])}); Iman-Davenport F = {f['f']:.2f} "
        f"({p_text(f['p_f'])}); {f['n_datasets']} datasets, {f['n_models']} "
        "models.\n"
    )
    out += ["| Model | " + cols + " | Average rank |"]
    out += ["|---" * (len(s["datasets"]) + 2) + "|"]
    for m in order:
        i = names.index(m)
        cells = [f"{s['ranks'][j, i]:g}" for j in range(len(s["datasets"]))]
        out.append(
            f"| {SHORT[m]} | " + " | ".join(cells) + f" | {s['avg_rank'][m]:.2f} |"
        )
    out.append("\n### Elo rating (Bradley-Terry)\n")
    out.append(
        f"{s['n_tasks']} dataset-by-repeat comparisons of {len(names)} models; "
        f"{ANCHOR} anchored at 1000; 95% percentile CI from {N_BOOT} bootstrap "
        "resamples of the datasets.\n"
    )
    out += ["| Model | Elo | 95% CI |", "|---|---|---|"]
    for m in s["elo_order"]:
        lo, hi = s["elo_ci"][m]
        out.append(f"| {SHORT[m]} | {s['elo'][m]:.0f} | [{lo:.0f}, {hi:.0f}] |")
    top = s["elo_order"][0]
    out.append(f"\n### Wilcoxon signed-rank tests against {SHORT[top]}\n")
    out += [
        "| Model | Datasets where it beats the top model | W | p | p (Holm) |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(s["wilcoxon"], key=lambda r: -s["elo"][names[r["model"]]]):
        out.append(
            f"| {SHORT[names[r['model']]]} | {r['beats']} of {len(s['datasets'])} "
            f"| {r['statistic']:g} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} |"
        )
    out.append("\n### Paired comparison with the best model (difference in RMSE)\n")
    out.append(
        "Cell: mean fold difference (model minus best), then the Holm-adjusted "
        "p-value across the other models on that dataset.\n"
    )
    out += [
        "| Model | " + cols + " |",
        "|---" * (len(s["datasets"]) + 1) + "|",
    ]
    for m in order:
        cells = []
        for d in s["datasets"]:
            t = s["table"][d]
            dec = decimals(results[d], t)
            if m == t["best"]:
                cells.append("best")
            else:
                r = t["paired"][m]
                cells.append(f"{r['diff']:+.{dec}f} ({p_text(r['p_holm'])})")
        out.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    out.append("\n#### Full paired tests per dataset\n")
    for d in s["datasets"]:
        t = s["table"][d]
        dec = decimals(results[d], t)
        info = results[d]
        out.append(
            f"**{d}** (n={info['n_samples']}, p={info['n_features']}, best: "
            f"{SHORT[t['best']]}, test/train ratio {t['ratio']:.3f})\n"
        )
        out += [
            "| Model | Mean diff | 95% corrected CI | t | p | p (Holm) |",
            "|---|---|---|---|---|---|",
        ]
        for m in order:
            if m == t["best"]:
                continue
            r = t["paired"][m]
            out.append(
                f"| {SHORT[m]} | {r['diff']:+.{dec}f} | [{r['low']:+.{dec}f}, "
                f"{r['high']:+.{dec}f}] | {r['t']:.2f} | {fmt_p(r['p'])} "
                f"| {fmt_p(r['p_holm'])} |"
            )
        out.append("")
    out.append("### RMSE spread: median and worst fold\n")
    out += [
        "| Model | " + cols + " |",
        "|---" * (len(s["datasets"]) + 1) + "|",
    ]
    for m in order:
        cells = []
        for d in s["datasets"]:
            v = np.asarray(results[d]["models"][m]["rmse"])
            dec = decimals(results[d], s["table"][d])
            cells.append(f"{np.median(v):.{dec}f} / {v.max():.{dec}f}")
        out.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    out.append("\nCells are median / worst fold RMSE.")
    for metric, label in (("mae", "MAE"), ("r2", "R²")):
        out.append(f"\n### Mean {label}\n")
        out += [
            "| Model | " + cols + " |",
            "|---" * (len(s["datasets"]) + 1) + "|",
        ]
        for m in order:
            cells = []
            for d in s["datasets"]:
                v = np.mean(results[d]["models"][m][metric])
                dec = 3 if metric == "r2" else decimals(results[d], s["table"][d])
                cells.append(f"{v:.{dec}f}")
            out.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    out.append("\n### Mean fit time / predict time per fold, seconds\n")
    out += [
        "| Model | " + cols + " |",
        "|---" * (len(s["datasets"]) + 1) + "|",
    ]
    for m in order:
        cells = []
        for d in s["datasets"]:
            r = results[d]["models"][m]
            cells.append(
                f"{np.mean(r['fit_seconds']):.3g} / {np.mean(r['predict_seconds']):.3g}"
            )
        out.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    out.append("\nFit time includes the inner grid search for the tuned models.")
    out.append("\nSettings: " + json.dumps(settings))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- figures

TITLE = {
    "yacht": "Yacht",
    "cars": "Auto MPG",
    "energy": "Energy efficiency",
    "concrete": "Concrete strength",
    "airfoil": "Airfoil self-noise",
    "ccpp": "Power plant",
    "california": "California housing",
}
XLABEL = {
    "yacht": "RMSE (unitless)",
    "cars": "RMSE (mpg)",
    "energy": "RMSE (load units)",
    "concrete": "RMSE (MPa)",
    "airfoil": "RMSE (dB)",
    "ccpp": "RMSE (MW)",
    "california": "RMSE ($100k, log scale)",
}


def is_lowess(name):
    return name.startswith("LOWESS")


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(RESULTS_DIR / f"{stem}.{ext}", dpi=plot_style.DPI)


def figure_rmse(results, s):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

    plot_style.apply()
    order, ds = s["order"], s["datasets"]
    fig, axes = plt.subplots(2, 4, figsize=(plot_style.WIDTH, 5.2), sharey=True)
    for k, d in enumerate(ds):
        ax = axes[k // 4, k % 4]
        info = results[d]
        data = [info["models"][m]["rmse"] for m in order]
        box = ax.boxplot(
            data,
            orientation="horizontal",
            positions=range(len(order)),
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            whis=1.5,
            boxprops={"edgecolor": "black", "linewidth": 0.7},
            whiskerprops={"color": "black", "linewidth": 0.7},
            capprops={"color": "black", "linewidth": 0.7},
            medianprops={"color": "black", "linewidth": 0.8},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": "none",
                "markeredgecolor": "black",
                "markeredgewidth": 0.6,
            },
            meanprops={
                "marker": "D",
                "markersize": 2.3,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },
        )
        for patch, m in zip(box["boxes"], order, strict=True):
            patch.set_facecolor("white" if is_lowess(m) else plot_style.BASELINE_FILL)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_yticks(range(len(order)), [SHORT[m] for m in order])
        ax.tick_params(axis="y", labelleft=k % 4 == 0)
        ax.set_title(
            f"{TITLE[info['key']]}\nn = {info['n_samples']}, p = {info['n_features']}"
        )
        ax.set_xlabel(XLABEL[info["key"]])
        if info["key"] == "california":
            ax.set_xscale("log")
            ticks = [0.5, 1, 2, 4]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[1, 3].remove()
    fig.subplots_adjust(
        left=0.205, right=0.99, top=0.9, bottom=0.09, wspace=0.12, hspace=0.55
    )
    save(fig, "figA1_rmse")
    plt.close(fig)


def figure_elo(results, s):
    """Figure 1: Elo leaderboard with bootstrap CIs, one column wide."""
    import matplotlib.pyplot as plt

    plot_style.apply()
    order = s["elo_order"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for row, m in enumerate(order):
        lo, hi = s["elo_ci"][m]
        elo = s["elo"][m]
        ax.plot([lo, hi], [row, row], color="black", lw=0.8, solid_capstyle="butt")
        ax.plot(
            [elo],
            [row],
            marker="o",
            markersize=4.5,
            markerfacecolor="black" if is_lowess(m) else "white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="none",
        )
        ax.annotate(
            f"{elo:.0f}",
            (1.0, row),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.axvline(1000, color="black", lw=0.4, ls=":")
    ax.set_yticks(range(len(order)), [SHORT[m] for m in order])
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xlabel("Elo rating (linear regression = 1000)")
    fig.subplots_adjust(left=0.38, right=0.88, top=0.97, bottom=0.17)
    save(fig, "fig1_elo")
    plt.close(fig)


def figure_time(results, s):
    """Figure 2: Elo against median fit time with the Pareto front."""
    import matplotlib.pyplot as plt

    plot_style.apply()
    order = s["order"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    times = {
        m: float(
            np.median(
                [
                    np.median(results[d]["models"][m]["fit_seconds"])
                    for d in s["datasets"]
                ]
            )
        )
        for m in order
    }
    elo = s["elo"]
    front = bs.pareto_front([times[m] for m in order], [elo[m] for m in order])
    fx = [times[order[i]] for i in front] + [20.0]
    fy = [elo[order[i]] for i in front]
    ax.step(fx, [*fy, fy[-1]], where="post", color="black", lw=0.6)
    place = {  # horizontal alignment and offset in points, by model
        "HistGradientBoosting": ("right", -5, 6),
        "Random forest": ("left", 5, 0),
        "Gradient boosting": ("right", -5, -4),
        "LOWESS + RF residuals": ("left", 5, 0),
        "LOWESS": ("left", 5, 0),
        "LOWESS boosting": ("center", 0, -9),
        "KNN": ("right", -5, 7),
        "Linear regression": ("left", 3, 8),
    }
    for m in order:
        ax.plot(
            [times[m]],
            [elo[m]],
            marker="o",
            markersize=4.5,
            markerfacecolor="black" if is_lowess(m) else "white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="none",
            zorder=3,
        )
        side, dx, dy = place[SHORT[m]]
        ax.annotate(
            SHORT[m],
            (times[m], elo[m]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=side,
            va="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 20)
    ax.set_ylim(940, 1760)
    ax.set_ylabel("Elo rating (linear regression = 1000)")
    ax.set_xlabel("Median fit time per fold (s)")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.15)
    save(fig, "fig2_time")
    plt.close(fig)


def main():
    saved = json.loads((RESULTS_DIR / "results.json").read_text())
    results, settings = saved["results"], saved["settings"]
    s = summarize(results, settings)
    md = markdown(results, settings, s)
    (RESULTS_DIR / "results.md").write_text(md)
    print(md)
    figure_rmse(results, s)
    figure_elo(results, s)
    figure_time(results, s)
    print("Saved figures to", RESULTS_DIR)


if __name__ == "__main__":
    main()
