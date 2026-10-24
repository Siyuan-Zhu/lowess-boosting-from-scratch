"""Tables and figures from ``results/results.json`` (no refitting).

    python benchmarks/report.py

Writes ``results/results.md`` plus ``fig1_rmse`` and ``fig2_rank`` as PNG
(200 dpi) and SVG. Needs matplotlib (the ``bench`` extra).

Per dataset, each model's mean RMSE gets a 95% Nadeau-Bengio corrected
interval, and each model is compared with the best model on paired fold
scores (same splits for every model). Across datasets the models are ranked
by mean RMSE, tested with Friedman's test, and drawn as a Nemenyi
critical-difference diagram when there are six or more datasets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bench_stats as bs
import numpy as np
import plot_style

RESULTS_DIR = Path(__file__).parent / "results"

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


def summarize(results):
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
    return {
        "names": names,
        "datasets": datasets,
        "table": table,
        "ranks": ranks,
        "avg_rank": dict(zip(names, avg, strict=True)),
        "order": [names[i] for i in np.argsort(avg, kind="stable")],
        "friedman": bs.friedman_test(means),
        "cd": bs.nemenyi_critical_difference(len(names), len(datasets)),
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
        f"models. Nemenyi critical difference at 0.05: {s['cd']:.2f} ranks.\n"
    )
    out += ["| Model | " + cols + " | Average rank |"]
    out += ["|---" * (len(s["datasets"]) + 2) + "|"]
    for m in order:
        i = names.index(m)
        cells = [f"{s['ranks'][j, i]:g}" for j in range(len(s["datasets"]))]
        out.append(
            f"| {SHORT[m]} | " + " | ".join(cells) + f" | {s['avg_rank'][m]:.2f} |"
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


def cd_cliques(avg_sorted, cd):
    """Maximal runs of models whose average ranks differ by less than ``cd``."""
    runs = []
    for i in range(len(avg_sorted)):
        j = i
        while j + 1 < len(avg_sorted) and avg_sorted[j + 1] - avg_sorted[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in runs):
            runs.append((i, j))
    return runs


def figure_rank(results, s):
    """Figure 1: Nemenyi critical-difference diagram, one panel."""
    import matplotlib.pyplot as plt

    plot_style.apply()
    order, k = s["order"], len(s["order"])
    avg = [s["avg_rank"][m] for m in order]
    fig = plt.figure(figsize=(plot_style.WIDTH, 2.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.set_xlim(-3.7, k + 3.4)
    ax.set_ylim(-3.55, 2.0)
    ax.plot([1, k], [0, 0], color="black", lw=0.6)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.15], color="black", lw=0.6)
        ax.text(r, 0.3, str(r), ha="center", va="bottom", fontsize=8)
    half = (k + 1) // 2
    for i, (m, a) in enumerate(zip(order, avg, strict=True)):
        left = i < half
        depth = -(0.95 + 0.7 * (i if left else k - 1 - i))
        xt = 0.55 if left else k + 0.45
        ax.plot([a, a, xt], [0, depth, depth], color="black", lw=0.6)
        ax.plot(
            [a],
            [0],
            marker="o",
            markersize=3.5,
            markerfacecolor="black" if is_lowess(m) else "white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=3,
        )
        ax.text(
            xt + (-0.12 if left else 0.12),
            depth,
            f"{SHORT[m]} ({a:.2f})",
            ha="right" if left else "left",
            va="center",
            fontsize=8,
        )
    for n, (i, j) in enumerate(cd_cliques(avg, s["cd"])):
        y = -0.4 - 0.2 * n
        ax.plot(
            [avg[i] - 0.05, avg[j] + 0.05],
            [y, y],
            color="black",
            lw=1.6,
            solid_capstyle="butt",
        )
    ax.plot([1, 1 + s["cd"]], [1.4, 1.4], color="black", lw=0.7)
    for x in (1, 1 + s["cd"]):
        ax.plot([x, x], [1.28, 1.52], color="black", lw=0.7)
    ax.text(1 + s["cd"] / 2, 1.63, f"CD = {s['cd']:.2f}", ha="center", fontsize=8)
    save(fig, "fig1_rank")
    plt.close(fig)


def figure_time(results, s):
    """Figure 2: average rank against median fit time, one column wide."""
    import matplotlib.pyplot as plt

    plot_style.apply()
    order, k = s["order"], len(s["order"])
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
    place = {  # horizontal alignment and offset in points, by model
        "HistGradientBoosting": ("right", -5, 5),
        "Random forest": ("left", 5, 0),
        "Gradient boosting": ("right", -5, -5),
        "LOWESS + RF residuals": ("left", 5, 0),
        "LOWESS": ("left", 5, 0),
        "LOWESS boosting": ("right", 0, -9),
        "KNN": ("left", 5, 0),
        "Linear regression": ("left", 5, 0),
    }
    for m in order:
        ax.plot(
            [times[m]],
            [s["avg_rank"][m]],
            marker="o",
            markersize=4.5,
            markerfacecolor="black" if is_lowess(m) else "white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="none",
        )
        side, dx, dy = place[SHORT[m]]
        ax.annotate(
            SHORT[m],
            (times[m], s["avg_rank"][m]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=side,
            va="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 20)
    ax.set_ylim(k + 0.4, 0.6)
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_xlabel("Median fit time per fold (s)")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.15)
    save(fig, "fig2_time")
    plt.close(fig)


def main():
    saved = json.loads((RESULTS_DIR / "results.json").read_text())
    results, settings = saved["results"], saved["settings"]
    s = summarize(results)
    md = markdown(results, settings, s)
    (RESULTS_DIR / "results.md").write_text(md)
    print(md)
    figure_rmse(results, s)
    figure_rank(results, s)
    figure_time(results, s)
    print("Saved figures to", RESULTS_DIR)


if __name__ == "__main__":
    main()
