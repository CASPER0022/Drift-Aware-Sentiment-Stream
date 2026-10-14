"""System architecture diagram for the report (analysis/figures/fig0_architecture.{png,pdf}).

Coordinates are in inches of the final 7.16 x 2.3 in figure (IEEE double-column width).
Usage:  python analysis/plot_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
W, H = 7.16, 2.3
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
FILL, EDGE = "#f4f7fc", "#9fb6d9"


def box(ax, x, y, w, h, title, lines=(), edge=EDGE, fill=FILL, title_size=7.5, gap=0.32):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=1.0, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.1, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    if lines:
        ax.text(x + w / 2, y + h - gap,"\n".join(lines), ha="center", va="top", fontsize=6.1,
                color=MUTED, linespacing=1.4)


def arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=INK, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, y + 0.06, label, ha="center", va="bottom", fontsize=5.6, color=MUTED)


def main() -> None:
    plt.rcParams.update({"font.family": "serif"})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    y, h, mid = 0.35, 1.55, 0.35 + 1.55 / 2
    # Spark group (dashed) around executors and driver
    ax.add_patch(FancyBboxPatch((2.23, 0.12), 2.9, 2.08, boxstyle="round,pad=0,rounding_size=0.08",
                                linewidth=0.8, edgecolor="#cfccc4", facecolor="#fbfaf8",
                                linestyle="--"))
    ax.text(3.68, 2.08, "Spark Structured Streaming", ha="center", va="center", fontsize=6.4,
            color=MUTED, style="italic")

    box(ax, 0.05, y, 0.92, h, "Producer", ["replays a stream", "in time order", "at --rate tweets/s",
                                            "as JSON"])
    box(ax, 1.22, y, 0.82, h, "Kafka", ["topic: tweets", "1 partition", "(keeps order)"])
    box(ax, 2.33, y, 1.02, h, "Executors", ["read from Kafka", "parse JSON", "tokenize (UDF)",
                                             "micro-batches", "of <= 1k tweets"])
    box(ax, 3.55, 0.22, 1.48, 1.78, "Driver: foreachBatch", ["per tweet: predict, then learn"])
    box(ax, 3.67, 0.90, 1.24, 0.58, "Baseline", ["ageing MNB +", "vocabulary detector"],
        edge=ORANGE, fill="white", title_size=6.8, gap=0.27)
    box(ax, 3.67, 0.28, 1.24, 0.58, "Enhanced", ["baseline +", "ADWIN on the error"],
        edge=BLUE, fill="white", title_size=6.8, gap=0.27)
    box(ax, 5.35, y, 0.82, h, "InfluxDB", ["per-model,", "pipeline and", "drift-event", "metrics"])
    box(ax, 6.33, y, 0.78, h, "Grafana", ["live charts", "+ drift", "annotations"])

    arrow(ax, 0.97, 1.22, mid)
    arrow(ax, 2.04, 2.33, mid)
    arrow(ax, 3.35, 3.55, mid)
    arrow(ax, 5.03, 5.35, mid)
    arrow(ax, 6.17, 6.33, mid)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig0_architecture.{ext}", dpi=200)
    print(f"wrote {OUT / 'fig0_architecture.png'}")


if __name__ == "__main__":
    main()
