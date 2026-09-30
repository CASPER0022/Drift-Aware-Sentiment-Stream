"""Baseline figures in the style of Iosifidis et al. Fig. 2 and Fig. 3.

  baseline_overall_accuracy.{png,pdf}   overall accuracy per strategy, DS1 and DS2 (Fig. 2)
  baseline_accuracy_over_time.{png,pdf} accuracy over time for accumulative, fading and the
                                        best informed strategy, with its detected changes
                                        and lambda trace (Fig. 3)

Reads experiments/results/baseline_*.csv written by experiments/run_baseline.py.
Usage:  python analysis/plot_baseline.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.streams import stream_path  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "analysis" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
REFERENCE = "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STREAM_TITLES = {"ds1": "DS1", "ds2": "DS2"}


def style(ax, grid_axis="y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def drift_points(stream: str) -> list[int]:
    path = stream_path(stream)
    return json.loads(path.with_name(f"{path.stem}_drift_points.json").read_text())["drift_points"]


def plot_overall(summary: pd.DataFrame) -> None:
    order = list(dict.fromkeys(summary["config"]))  # config order as in baseline.yaml
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), sharey=True, constrained_layout=True)
    for ax, (stream, rows) in zip(axes, summary.groupby("stream", sort=False)):
        rows = rows.set_index("config").loc[order]
        y = range(len(order))[::-1]
        colors = [REFERENCE if m != "informed" else BLUE if c.endswith("-Init") else ORANGE
                  for c, m in zip(rows.index, rows["model"])]
        ax.hlines(y, rows["accuracy"].min() - 0.01, rows["accuracy"], color=GRID, linewidth=0.8)
        ax.scatter(rows["accuracy"], y, color=colors, s=36, zorder=3,
                   edgecolor="white", linewidth=1.2)
        for yi, acc in zip(y, rows["accuracy"]):
            ax.annotate(f"{acc:.1%}", (acc, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=MUTED)
        ax.set_yticks(list(y), order, fontsize=7)
        ax.set_xlim(rows["accuracy"].min() - 0.01, rows["accuracy"].max() + 0.012)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(STREAM_TITLES[stream], loc="left", fontsize=9)
        ax.set_xlabel("overall prequential accuracy", fontsize=7.5)
        style(ax, grid_axis="x")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=l, markersize=6)
               for c, l in ((REFERENCE, "reference model"), (BLUE, "informed, lambda0 > 0 (Init)"),
                            (ORANGE, "informed, lambda0 = 0 (Zero)"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    save(fig, "baseline_overall_accuracy")


def plot_over_time(summary, windows, events, lambdas) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.2), height_ratios=[3, 1.2], sharex="col",
                             constrained_layout=True)
    best = {}
    for col, stream in enumerate(["ds1", "ds2"]):
        rows = summary[summary["stream"] == stream]
        best_cfg = rows[rows["model"] == "informed"].sort_values("accuracy").iloc[-1]["config"]
        best[stream] = best_cfg
        ax, lax = axes[0, col], axes[1, col]
        for p in drift_points(stream):
            for a in (ax, lax):
                a.axvline(p, color=MUTED, linestyle="--", linewidth=0.7, zorder=1)
        for cfg, color in (("accumulativeMNB", ORANGE), ("fadingMNB", AQUA), (best_cfg, BLUE)):
            w = windows[(windows["stream"] == stream) & (windows["config"] == cfg)]
            ax.plot(w["end_idx"], w["accuracy"], color=color, linewidth=1.1, label=cfg, zorder=2)
        changes = events[(events["stream"] == stream) & (events["config"] == best_cfg)
                         & (events["kind"] == "change")]["idx"]
        ax.scatter(changes, [1.02] * len(changes), marker="v", s=22, color=BLUE, zorder=3,
                   clip_on=False, label=f"change detected ({best_cfg})")
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"{STREAM_TITLES[stream]} (dashed = true drift point)", loc="left", fontsize=8.5,
                     pad=14)
        if col == 0:
            ax.set_ylabel("accuracy per 10k window", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        style(ax)

        lam = lambdas[(lambdas["stream"] == stream) & (lambdas["config"] == best_cfg)]
        n = windows[windows["stream"] == stream]["end_idx"].max()
        lax.step(list(lam["idx"]) + [n], list(lam["lambda"]) + [lam["lambda"].iloc[-1]],
                 where="post", color=BLUE, linewidth=1.1)
        lax.set_ylim(0, max(0.55, lam["lambda"].max() * 1.1))
        lax.set_xlim(0, n)
        if col == 0:
            lax.set_ylabel("lambda", fontsize=7.5)
        lax.set_xlabel("instance index", fontsize=7.5)
        lax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
        style(lax)
    save(fig, "baseline_accuracy_over_time")
    return best


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/{name}.png")


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK})
    summary = pd.read_csv(RESULTS / "baseline_summary.csv")
    windows = pd.read_csv(RESULTS / "baseline_windows.csv")
    events = pd.read_csv(RESULTS / "baseline_events.csv")
    lambdas = pd.read_csv(RESULTS / "baseline_lambda.csv")
    plot_overall(summary)
    best = plot_over_time(summary, windows, events, lambdas)
    print(f"best informed strategy: {best}")


if __name__ == "__main__":
    main()
