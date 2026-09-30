"""Sanity plots for every stream: windowed class/topic mix with the true drift points.

Writes analysis/figures/streams/<stream>.png for DS1, DS2 and each scenario (seed 0 by
default), plus an overview grid streams_overview.{png,pdf}.

Usage:  python analysis/plot_streams.py [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from streams import PROCESSED, SCENARIOS  # noqa: E402

FIG_DIR = ROOT / "analysis" / "figures"
POS, TOPIC_B, FLIPPED = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
MUTED, GRID, SHADE = "#52514e", "#e4e3df", "#f0efec"
SCENARIO_NAMES = ["s_sudden", "s_gradual", "s_recurring", "s_label_flip", "s_prior_shift"]


def windowed(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values) // window
    return (np.arange(n) + 0.5) * window, values[: n * window].reshape(n, window).mean(axis=1)


def draw(ax, df: pd.DataFrame, meta: dict) -> None:
    window = 10_000 if len(df) > 500_000 else 1_000
    for start, end in meta.get("drift_windows", []):
        ax.axvspan(start, end, color=SHADE, zorder=0)
    for p in meta["drift_points"]:
        ax.axvline(p, color=MUTED, linestyle="--", linewidth=0.8)

    ax.plot(*windowed(df["label"].to_numpy(), window), color=POS, linewidth=1.2, label="positive")
    if "topic" in df and (df["topic"] == "B").any():
        ax.plot(*windowed((df["topic"] == "B").to_numpy(), window), color=TOPIC_B,
                linewidth=1.2, label="topic B (music/movies)")
    flipped = (df["label"] != df["orig_label"]).to_numpy() if "orig_label" in df else None
    if flipped is not None and flipped.any():
        x, flip_share = windowed(flipped, window)
        _, a_share = windowed((df["topic"] == "A").to_numpy(), window)
        ax.plot(x, flip_share / np.maximum(a_share, 1e-9), color=FLIPPED, linewidth=1.2,
                label="topic A labels flipped")

    ax.set_title(f"{meta['stream']}: {meta['drift_type']}", loc="left", fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, len(df))
    ax.set_ylabel(f"share per {window:,}", fontsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1e6:.1f}M" if len(df) > 500_000 else f"{v / 1e3:.0f}k"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", bbox_to_anchor=(1, 1),
              ncol=3, borderaxespad=0.2, handlelength=1.5)


def load(path: Path) -> tuple[pd.DataFrame, dict]:
    meta = json.loads(path.with_name(f"{path.stem}_drift_points.json").read_text())
    cols = ["label", "topic", "orig_label"] if "seed" in path.stem else ["label"]
    return pd.read_parquet(path, columns=cols), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = [PROCESSED / "ds1.parquet", PROCESSED / "ds2.parquet"] + \
            [SCENARIOS / f"{name}_seed{args.seed}.parquet" for name in SCENARIO_NAMES]
    streams = [load(p) for p in paths]

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    out_dir = FIG_DIR / "streams"
    out_dir.mkdir(parents=True, exist_ok=True)
    for df, meta in streams:
        fig, ax = plt.subplots(figsize=(7.16, 2.0), constrained_layout=True)
        draw(ax, df, meta)
        ax.set_xlabel("instance index", fontsize=7)
        fig.savefig(out_dir / f"{meta['stream']}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(len(streams), 1, figsize=(7.16, 1.55 * len(streams)),
                             constrained_layout=True)
    for ax, (df, meta) in zip(axes, streams):
        draw(ax, df, meta)
    axes[-1].set_xlabel("instance index (dashed = true drift point, shaded = gradual window)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"streams_overview.{ext}", dpi=200)
    print(f"wrote {len(streams)} stream plots to {out_dir.relative_to(ROOT)} "
          f"and {FIG_DIR.relative_to(ROOT)}/streams_overview.png")


if __name__ == "__main__":
    main()
