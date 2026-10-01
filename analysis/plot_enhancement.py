"""Headline figure for the ADWIN enhancement (reads results/enhancement_*.csv).

  (a) S-label-flip: accuracy over time, baseline (vocabulary detector) vs enhanced (+ ADWIN),
      with each model's detections
  (b) the same, zoomed on the drift
  (c) detection delay of each detector on every stream (log scale)

Usage:  python analysis/plot_enhancement.py [--enhanced enhanced-or]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "analysis" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # enhanced / baseline (categorical slots 1-2)
REFERENCE = "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STREAM_LABELS = {"s_label_flip_seed0": "S-label-flip", "s_label_flip_topic_seed0": "S-label-flip\n(topic only)",
                 "ds1": "DS1"}


def style(ax, grid_axis="y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def accuracy_panel(ax, windows, events, stream, enhanced, drift, xlim=None) -> None:
    ax.axvline(drift, color=MUTED, linestyle="--", linewidth=0.8)
    series = (("accumulativeMNB", REFERENCE, "accumulative MNB"),
              ("baseline", ORANGE, "baseline (vocabulary detector)"),
              (enhanced, BLUE, f"enhanced (+ ADWIN, {enhanced.split('-', 1)[1]})"))
    for cfg, color, label in series:
        w = windows[(windows["stream"] == stream) & (windows["config"] == cfg)]
        ax.plot(w["end_idx"], w["accuracy"], color=color, linewidth=1.2, label=label)
    for cfg, det, marker, color in (("baseline", "vocab", "v", ORANGE), (enhanced, "adwin", "^", BLUE)):
        idx = events[(events["stream"] == stream) & (events["config"] == cfg)
                     & (events["detector"] == det) & (events["kind"] == "change")]["idx"]
        if xlim:
            idx = idx[(idx >= xlim[0]) & (idx <= xlim[1])]
        ax.scatter(idx, [1.03] * len(idx), marker=marker, s=26, color=color, clip_on=False,
                   zorder=4, label=f"{'vocabulary' if det == 'vocab' else 'ADWIN'} detection")
    ax.set_ylim(0.2, 1.0)
    if xlim:
        ax.set_xlim(*xlim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    style(ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enhanced", default="enhanced-or", help="enhanced config to show in (a), (b)")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK})
    summary = pd.read_csv(RESULTS / "enhancement_summary.csv")
    windows = pd.read_csv(RESULTS / "enhancement_windows.csv")
    events = pd.read_csv(RESULTS / "enhancement_events.csv")
    stream = "s_label_flip_seed0"
    drift = int(summary[summary["stream"] == stream]["drift_point"].iloc[0])

    fig = plt.figure(figsize=(7.16, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    accuracy_panel(ax_a, windows, events, stream, args.enhanced, drift)
    ax_a.set_title("(a) S-label-flip: every label inverts at 40k, the words stay the same",
                   loc="left", fontsize=8.5, pad=12)
    ax_a.set_ylabel("accuracy per 500 tweets", fontsize=7.5)
    ax_a.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)

    accuracy_panel(ax_b, windows, events, stream, args.enhanced, drift, xlim=(36_000, 50_000))
    ax_b.set_title("(b) zoom on the drift", loc="left", fontsize=8.5, pad=12)
    ax_b.set_xlabel("instance index", fontsize=7.5)

    rows = summary[summary["config"] == args.enhanced].set_index("stream")
    base = summary[summary["config"] == "baseline"].set_index("stream")
    streams = [s for s in STREAM_LABELS if s in rows.index]
    y = np.arange(len(streams))
    for offset, delays, color, label in ((0.18, base["vocab_mean_delay"], ORANGE, "vocabulary detector"),
                                         (-0.18, rows["adwin_mean_delay"], BLUE, "ADWIN")):
        for yi, s in zip(y, streams):
            d = pd.to_numeric(delays.get(s), errors="coerce")
            if np.isnan(d):
                ax_c.annotate("not detected", (1.2, yi + offset), fontsize=6.5, color=color,
                              va="center")
            else:
                ax_c.barh(yi + offset, max(d, 1), height=0.32, color=color,
                          label=label if yi == 0 else None)
                ax_c.annotate(f"{int(d):,}", (max(d, 1), yi + offset), xytext=(3, 0),
                              textcoords="offset points", va="center", fontsize=6.5, color=MUTED)
    ax_c.set_xscale("log")
    ax_c.set_xlim(1, 2e5)
    ax_c.set_yticks(y, [STREAM_LABELS[s] for s in streams], fontsize=7)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("instances after the true drift (log scale)", fontsize=7.5)
    ax_c.set_title("(c) detection delay", loc="left", fontsize=8.5, pad=12)
    ax_c.legend(frameon=False, fontsize=6.5, loc="upper right")
    style(ax_c, grid_axis="x")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"enhancement_label_flip.{ext}", dpi=200)
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/enhancement_label_flip.png")


if __name__ == "__main__":
    main()
