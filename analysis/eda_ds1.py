"""EDA for DS1: reproduce Iosifidis et al. Fig. 1a and locate the natural change point.

Writes analysis/figures/ds1_hourly_class_distribution.{png,pdf} and prints summary facts.

Usage:  python analysis/eda_ds1.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS1 = ROOT / "data" / "processed" / "ds1.parquet"
FIG_DIR = ROOT / "analysis" / "figures"

POS, NEG = "#2a78d6", "#eb6834"  # categorical slots 1-2 (CVD-validated pair)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHARE_WINDOW = 10_000  # instances per point in the positive-share panel


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    df = pd.read_parquet(DS1, columns=["idx", "ts", "label"])
    labels = df["label"].to_numpy()

    last_pos = int(np.flatnonzero(labels == 1).max())
    change_idx = last_pos + 1  # from here on the stream is 100% negative
    change_ts = df["ts"].iat[change_idx]

    # Hourly counts per class on a continuous hourly grid, so collection gaps show as zeros.
    hourly = (df.set_index("ts").groupby("label").resample("1h").size()
              .unstack(0, fill_value=0).rename(columns={0: "neg", 1: "pos"}))
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h"),
                            fill_value=0)

    n_win = len(labels) // SHARE_WINDOW
    share = labels[: n_win * SHARE_WINDOW].reshape(n_win, SHARE_WINDOW).mean(axis=1)
    share_x = (np.arange(n_win) + 0.5) * SHARE_WINDOW

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.6), constrained_layout=True)

    ax1.plot(hourly.index, hourly["pos"], color=POS, linewidth=1.0, label="positive")
    ax1.plot(hourly.index, hourly["neg"], color=NEG, linewidth=1.0, label="negative")
    ax1.axvline(change_ts, color=MUTED, linestyle="--", linewidth=0.8)
    ax1.annotate("negative-only tail", xy=(change_ts, ax1.get_ylim()[1] * 0.92),
                 xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED)
    ax1.set_ylabel("tweets per hour")
    ax1.set_title("(a) Hourly class distribution (collection gaps shown as zero)",
                  loc="left", fontsize=8.5)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.legend(frameon=False, fontsize=7, loc="upper left")
    style_axes(ax1)

    ax2.plot(share_x, share, color=POS, linewidth=1.2)
    ax2.axhline(0.5, color=MUTED, linewidth=0.6, linestyle=":")
    ax2.axvline(change_idx, color=MUTED, linestyle="--", linewidth=0.8)
    ax2.annotate(f"change point\ninstance {change_idx:,}", xy=(change_idx, 0.8),
                 xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=7,
                 color=MUTED)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlim(0, len(labels))
    ax2.set_xlabel("instance index (arrival order)")
    ax2.set_ylabel("share positive")
    ax2.set_title(f"(b) Positive share per {SHARE_WINDOW:,}-tweet window",
                  loc="left", fontsize=8.5)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    style_axes(ax2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ds1_hourly_class_distribution.{ext}", dpi=200)

    active_hours = int((hourly.sum(axis=1) > 0).sum())
    print(f"tweets              : {len(df):,} ({labels.mean():.1%} positive)")
    print(f"span                : {df['ts'].min()} -> {df['ts'].max()}")
    print(f"hours with data     : {active_hours:,} of {len(hourly):,}")
    print(f"change point        : instance {change_idx:,} at {change_ts} "
          f"(paper: ~1,326,000)")
    print(f"negative-only tail  : {len(df) - change_idx:,} tweets")
    print(f"figure              : {FIG_DIR.relative_to(ROOT)}/ds1_hourly_class_distribution.png")


if __name__ == "__main__":
    main()
