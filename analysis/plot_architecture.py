"""System architecture diagram (analysis/figures/fig0_architecture.{png,pdf}).

Shows every tool of the pipeline with its version and the data flow between them.
Coordinates are in inches of the final 7.16 x 2.75 in figure (IEEE double-column width).
Usage:  python analysis/plot_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
W, H = 7.16, 2.75
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
FILL, EDGE = "#f4f7fc", "#9fb6d9"


def box(ax, x, y, w, h, title, lines=(), edge=EDGE, fill=FILL, title_size=6.9, gap=0.42,
        tool=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=1.0, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    if tool:
        ax.text(x + w / 2, y + h - 0.24, tool, ha="center", va="top", fontsize=5.4,
                color=INK, style="italic")
    if lines:
        ax.text(x + w / 2, y + h - gap, "\n".join(lines), ha="center", va="top", fontsize=5.6,
                color=MUTED, linespacing=1.35)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.0, color=INK, shrinkA=0, shrinkB=0))


def main() -> None:
    plt.rcParams.update({"font.family": "serif"})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    y, h = 0.55, 1.65
    mid = y + h / 2

    # Docker Compose boundary (Kafka ... Grafana) and the Spark group inside it
    ax.add_patch(FancyBboxPatch((1.86, 0.08), 5.25, 2.6, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0.7, edgecolor="#b9b5ab", facecolor="none", linestyle=":"))
    ax.text(4.48, 0.16, "Docker Compose (Docker 28.3): every service from Kafka to Grafana",
            ha="center", va="bottom", fontsize=5.6, color=MUTED, style="italic")
    ax.add_patch(FancyBboxPatch((2.78, 0.36), 2.52, 2.2, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0.8, edgecolor="#cfccc4", facecolor="#fbfaf8",
                                linestyle="--"))
    ax.text(4.04, 2.46, "Apache Spark 3.5.3 Structured Streaming", ha="center", va="center",
            fontsize=6.0, color=MUTED, style="italic")

    box(ax, 0.04, y, 0.8, h, "Data prep", ["Sentiment140", "1.6M tweets", "sorted by time;",
                                            "DS2 + drift", "scenarios", "-> parquet"],
        tool="pandas, pyarrow")
    box(ax, 0.97, y, 0.8, h, "Producer", ["replays a", "stream in", "time order at", "--rate",
                                           "tweets/s", "(JSON)"], tool="confluent-kafka")
    box(ax, 1.95, y, 0.72, h, "Kafka", ["topic", "tweets,", "1 partition", "(keeps", "order)"],
        tool="3.9.1, KRaft")
    box(ax, 2.88, y, 0.82, h, "Executors", ["read Kafka", "parse JSON", "tokenize", "(UDF)",
                                             "micro-batch", "<= 1k tweets"])
    box(ax, 3.82, 0.45, 1.4, 1.87, "Driver", ["foreachBatch: per tweet", "predict, then learn"],
        gap=0.25)
    box(ax, 3.92, 1.11, 1.2, 0.58, "Baseline", ["ageing MNB +", "vocabulary detector"],
        edge=ORANGE, fill="white", title_size=6.3, gap=0.25)
    box(ax, 3.92, 0.52, 1.2, 0.54, "Enhanced", ["baseline + ADWIN", "on the error (River)"],
        edge=BLUE, fill="white", title_size=6.3, gap=0.23)
    box(ax, 5.44, y, 0.78, h, "InfluxDB", ["per-model,", "pipeline and", "drift-event", "metrics"],
        tool="2.7")
    box(ax, 6.32, y, 0.76, h, "Grafana", ["live", "dashboard", "+ drift", "annotations"],
        tool="OSS 11.5.2")

    arrow(ax, 0.84, 0.97, mid)
    arrow(ax, 1.77, 1.95, mid)
    arrow(ax, 2.67, 2.88, mid)
    arrow(ax, 3.70, 3.82, mid)
    arrow(ax, 5.22, 5.44, mid)
    arrow(ax, 6.22, 6.32, mid)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig0_architecture.{ext}", dpi=250)
    print(f"wrote {OUT / 'fig0_architecture.png'}")


if __name__ == "__main__":
    main()
