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

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"
BLUE, GREY = "#2a78d6", "#898781"


def is_lowess(name):
    return name.startswith("LOWESS")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"{stem}.{ext}", dpi=200, facecolor=SURFACE)


def unit_text(units):
    return " (dimensionless)" if units == "dimensionless" else f" in {units}"


def figure_rmse(results, s):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    order = s["order"]
    ds = s["datasets"]
    ncols = 4
    nrows = math.ceil((len(ds) + 1) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 3.7 * nrows), facecolor=SURFACE, squeeze=False
    )
    rng = np.random.default_rng(0)
    for k, d in enumerate(ds):
        ax = axes[k // ncols, k % ncols]
        style_axes(ax)
        t = s["table"][d]
        info = results[d]
        cis = [t["ci"][m] for m in order]
        cap = None
        top = max(ci[2] for ci in cis)
        folds_all = [np.asarray(info["models"][m]["rmse"]) for m in order]
        if max(f.max() for f in folds_all) > 1.5 * top:
            cap = 1.1 * top
        for row, (m, folds) in enumerate(zip(order, folds_all, strict=True)):
            color = BLUE if is_lowess(m) else GREY
            shown = folds if cap is None else folds[folds <= cap]
            ax.scatter(
                shown,
                row + rng.uniform(-0.18, 0.18, shown.size),
                s=11,
                color=color,
                alpha=0.3,
                lw=0,
                zorder=2,
            )
            if cap is not None and (folds > cap).any():
                ax.scatter(
                    [cap] * int((folds > cap).sum()),
                    [row] * int((folds > cap).sum()),
                    marker=">",
                    s=26,
                    color=color,
                    zorder=3,
                    clip_on=False,
                )
            mean, lo, hi = t["ci"][m]
            ax.plot(
                [lo, hi],
                [row, row],
                color=color,
                lw=2.2,
                solid_capstyle="butt",
                zorder=3,
            )
            ax.scatter(
                [mean], [row], s=46, color=color, edgecolor=SURFACE, lw=1.5, zorder=4
            )
        ax.set_ylim(len(order) - 0.4, -0.7)
        ax.set_yticks(range(len(order)))
        if k % ncols == 0:
            ax.set_yticklabels([SHORT[m] for m in order], fontsize=9, color=INK)
            for label, m in zip(ax.get_yticklabels(), order, strict=True):
                if is_lowess(m):
                    label.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        if cap is not None:
            ax.set_xlim(0, cap)
            n_out = sum(int((f > cap).sum()) for f in folds_all)
            worst = max(f.max() for f in folds_all)
            note = (
                f"Axis cut at {cap:.1f}: {n_out} fold scores lie beyond it "
                f"(arrows; largest {worst:.2f}).\nIntervals are clipped at the "
                "edges of the axis."
            )
            ax.set_xlabel(note, fontsize=7.5, color=MUTED, labelpad=6)
        else:
            lo_all = min(min(f.min() for f in folds_all), min(ci[1] for ci in cis))
            hi_all = max(max(f.max() for f in folds_all), top)
            pad = 0.04 * (hi_all - lo_all)
            ax.set_xlim(max(0, lo_all - pad), hi_all + pad)
        ax.set_title(
            f"{d}\n",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
            pad=14,
        )
        ax.text(
            0,
            1.03,
            f"RMSE{unit_text(info['units'])}  ·  n={info['n_samples']}, "
            f"p={info['n_features']}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED,
            va="bottom",
        )
    # legend cell
    ax = axes[len(ds) // ncols, len(ds) % ncols]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.92
    ax.text(0, y, "How to read", fontsize=10, fontweight="bold", color=INK, va="top")
    ax.scatter([0.04], [0.76], s=46, color=BLUE, edgecolor=SURFACE, lw=1.5)
    ax.plot([0.0, 0.08], [0.76, 0.76], color=BLUE, lw=2.2, solid_capstyle="butt")
    ax.text(
        0.14,
        0.76,
        "mean RMSE and 95% corrected CI",
        fontsize=8.5,
        color=INK,
        va="center",
    )
    ax.scatter(
        [0.02, 0.04, 0.06], [0.62, 0.64, 0.61], s=11, color=BLUE, alpha=0.3, lw=0
    )
    ax.text(
        0.14, 0.62, "one test fold (15 per model)", fontsize=8.5, color=INK, va="center"
    )
    ax.scatter([0.04], [0.48], s=46, color=BLUE, edgecolor=SURFACE, lw=1.5)
    ax.text(
        0.14,
        0.48,
        "blue: LOWESS family (bold labels)",
        fontsize=8.5,
        color=INK,
        va="center",
    )
    ax.scatter([0.04], [0.34], s=46, color=GREY, edgecolor=SURFACE, lw=1.5)
    ax.text(0.14, 0.34, "grey: baselines", fontsize=8.5, color=INK, va="center")
    ax.text(
        0,
        0.18,
        (
            "Rows are sorted by average rank across\n"
            "datasets, best at the top. Lower is\n"
            "better. Each panel has its own x-axis."
        ),
        fontsize=8.5,
        color=MUTED,
        va="top",
    )
    fig.tight_layout(h_pad=2.2, w_pad=1.2)
    save(fig, "fig1_rmse")
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
    import matplotlib.pyplot as plt

    order, k = s["order"], len(s["order"])
    avg = [s["avg_rank"][m] for m in order]
    fig, (ax_cd, ax_t) = plt.subplots(
        1,
        2,
        figsize=(14, 4.6),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    # (a) critical-difference diagram, rank 1 on the left
    ax_cd.set_facecolor(SURFACE)
    ax_cd.axis("off")
    ax_cd.set_xlim(-3.4, k + 4.4)
    ax_cd.set_ylim(-6.6, 3.0)
    ax_cd.plot([1, k], [0, 0], color=MUTED, lw=1)
    for r in range(1, k + 1):
        ax_cd.plot([r, r], [0, 0.18], color=MUTED, lw=1)
        ax_cd.text(r, 0.4, str(r), ha="center", va="bottom", fontsize=8.5, color=MUTED)
    half = (k + 1) // 2
    for i, (m, a) in enumerate(zip(order, avg, strict=True)):
        color = BLUE if is_lowess(m) else GREY
        left = i < half
        depth = -(1.3 + 0.8 * (i if left else k - 1 - i))
        xt = 0.4 if left else k + 0.6
        ax_cd.plot([a, a], [0, depth], color=color, lw=1.4)
        ax_cd.plot([a, xt], [depth, depth], color=color, lw=1.4)
        ax_cd.scatter([a], [0], s=34, color=color, zorder=3)
        ax_cd.text(
            xt + (-0.12 if left else 0.12),
            depth,
            f"{SHORT[m]}  ({a:.2f})",
            ha="right" if left else "left",
            va="center",
            fontsize=9,
            color=INK,
        )
    for n, (i, j) in enumerate(cd_cliques(avg, s["cd"])):
        y = -0.45 - 0.22 * n
        ax_cd.plot(
            [avg[i] - 0.05, avg[j] + 0.05],
            [y, y],
            color=INK,
            lw=3.2,
            solid_capstyle="butt",
            zorder=4,
        )
    ax_cd.plot([1, 1 + s["cd"]], [1.7, 1.7], color=INK, lw=1.4)
    for x in (1, 1 + s["cd"]):
        ax_cd.plot([x, x], [1.58, 1.82], color=INK, lw=1.4)
    ax_cd.text(
        1 + s["cd"] / 2, 1.95, f"CD = {s['cd']:.2f}", ha="center", fontsize=9, color=INK
    )
    ax_cd.set_title(
        "(a) Average rank across datasets (1 = best)",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    # (b) accuracy against fit time
    style_axes(ax_t)
    ax_t.grid(axis="y", color=GRID, lw=0.6)
    times = {
        m: math.exp(
            np.mean(
                [
                    math.log(np.mean(results[d]["models"][m]["fit_seconds"]))
                    for d in s["datasets"]
                ]
            )
        )
        for m in order
    }
    offsets = {"HistGradientBoosting": (-8, 0, "right")}
    for m in order:
        color = BLUE if is_lowess(m) else GREY
        ax_t.scatter(
            [times[m]],
            [s["avg_rank"][m]],
            s=60,
            color=color,
            edgecolor=SURFACE,
            lw=1.5,
            zorder=3,
        )
    ax_t.set_xscale("log")
    ax_t.set_ylim(k + 0.5, 0.5)
    ax_t.set_ylabel("Average rank (1 = best)", color=MUTED, fontsize=9)
    ax_t.set_xlabel(
        "Fit time per fold, geometric mean over datasets (s, log scale)",
        color=MUTED,
        fontsize=9,
    )
    ax_t.set_title(
        "(b) Accuracy against fit time",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax_t.spines["left"].set_visible(True)
    ax_t.spines["left"].set_color(MUTED)
    ax_t.figure.canvas.draw()
    for m in order:
        dx, dy, ha = offsets.get(m, (8, 0, "left"))
        ax_t.annotate(
            SHORT[m],
            (times[m], s["avg_rank"][m]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            color=INK,
        )
    fig.tight_layout(w_pad=3)
    save(fig, "fig2_rank")
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
    print("Saved figures to", RESULTS_DIR)


if __name__ == "__main__":
    main()
