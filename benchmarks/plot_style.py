"""Matplotlib style shared by the benchmark figures (two-column journal width)."""

from __future__ import annotations

import matplotlib as mpl

WIDTH = 7.0  # inches, two-column page width
BASELINE_FILL = "0.85"  # light gray box fill for baselines; LOWESS family is white
DPI = 300


def apply():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "font.weight": "normal",
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
