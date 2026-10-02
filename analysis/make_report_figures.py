"""Day 11: every report figure and table from the Day 10 grid (experiments/results/grid*.csv).

Figures (analysis/figures/, PNG + PDF):
  fig1_accuracy_real.*        accuracy over time on DS1 / DS2 with detections (paper Fig. 3 style)
  fig2_label_flip_seeds.*     S-label-flip, mean and min-max band over 5 seeds
  fig3_detection.*            detection delay, false alarms and missed drifts per stream
  fig4_recovery.*             recovery time per stream
  fig5_ablation.*             fusion modes: accuracy gain over the baseline and detection counts
Tables (experiments/results/ as CSV, report/tables/ as LaTeX):
  table_accuracy              accuracy per stream x model, mean +- std over seeds
  table_main                  per model: accuracy, P/R/F1, detection, recovery, cost
  table_cost                  offline and streaming cost, baseline vs enhanced

Usage:  python analysis/make_report_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.streams import stream_path  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "analysis" / "figures"
TEX_DIR = ROOT / "report" / "tables"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"enhanced-or": BLUE, "baseline-Rebuild": ORANGE, "enhanced-adwin_only": AQUA,
                "enhanced-and": YELLOW, "accumulativeMNB": GRAY}
MODEL_LABELS = {"accumulativeMNB": "accumulative MNB", "fadingMNB": "fading MNB",
                "baseline-Rebuild": "baseline: Rebuild-Zero",
                "baseline-SIFR": "baseline: SlowIncreaseFastReset-Zero",
                "enhanced-adwin_only": "enhanced: ADWIN only", "enhanced-or": "enhanced: OR",
                "enhanced-and": "enhanced: AND"}
MODEL_ORDER = list(MODEL_LABELS)
STREAM_LABELS = {"ds1": "DS1", "ds2": "DS2", "s_label_flip": "label flip",
                 "s_label_flip_topic": "label flip (topic)", "s_prior_shift": "prior shift",
                 "s_sudden": "sudden", "s_gradual": "gradual", "s_recurring": "recurring"}
STREAM_ORDER = list(STREAM_LABELS)


# -- helpers ----------------------------------------------------------------------------

def style(ax, grid_axis="y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote analysis/figures/{name}.png")


def drift_points(stream: str) -> list[int]:
    path = stream_path(stream, 0)
    return json.loads(path.with_name(f"{path.stem}_drift_points.json").read_text())["drift_points"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def write_tex(df: pd.DataFrame, name: str, caption: str, label: str, col_format: str,
              wide: bool = False) -> None:
    """Minimal booktabs table (no jinja2 dependency); cells are already formatted strings.
    wide=True spans both columns of the IEEE two-column layout (table*)."""
    env = "table*" if wide else "table"
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")  # noqa: E731
    lines = [rf"\begin{{{env}}}[t]", r"\centering", r"\footnotesize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{col_format}}}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    (TEX_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote report/tables/{name}.tex")


# -- figures ----------------------------------------------------------------------------

def fig_accuracy_real(windows, events) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True, constrained_layout=True)
    for ax, stream in zip(axes, ["ds1", "ds2"]):
        for p in drift_points(stream):
            ax.axvline(p, color=MUTED, linestyle="--", linewidth=0.7)
        for model in ("accumulativeMNB", "baseline-Rebuild", "enhanced-or"):
            w = windows[(windows.stream == stream) & (windows.model == model)]
            ax.plot(w.end_idx, w.accuracy, color=MODEL_COLORS[model], linewidth=1.1,
                    label=MODEL_LABELS[model])
        for model, det, marker in (("baseline-Rebuild", "vocab", "v"), ("enhanced-or", "adwin", "^")):
            idx = events[(events.stream == stream) & (events.model == model) & (events.detector == det)
                         & (events.kind == "change") & (events.acted == True)]["idx"]  # noqa: E712
            ax.scatter(idx, [1.02] * len(idx), marker=marker, s=20, color=MODEL_COLORS[model],
                       clip_on=False, zorder=4,
                       label=f"{'vocabulary' if det == 'vocab' else 'ADWIN'} detection")
        ax.set_ylim(0.5, 1.0)
        ax.set_xlim(0, windows[windows.stream == stream].end_idx.max())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
        ax.set_title(f"{STREAM_LABELS[stream]} (dashed: true drift points)", loc="left",
                     fontsize=8.5, pad=10)
        ax.set_xlabel("instance index", fontsize=7.5)
        style(ax)
    axes[0].set_ylabel("accuracy per 10k tweets", fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower left")
    save(fig, "fig1_accuracy_real")


def fig_label_flip_seeds(windows) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.6), constrained_layout=True)
    stream = "s_label_flip"
    ax.axvline(drift_points(stream)[0], color=MUTED, linestyle="--", linewidth=0.7)
    for model in ("accumulativeMNB", "baseline-Rebuild", "enhanced-or"):
        w = windows[(windows.stream == stream) & (windows.model == model)]
        g = w.groupby("end_idx").accuracy
        ax.fill_between(g.min().index, g.min(), g.max(), color=MODEL_COLORS[model], alpha=0.18,
                        linewidth=0)
        ax.plot(g.mean().index, g.mean(), color=MODEL_COLORS[model], linewidth=1.2,
                label=MODEL_LABELS[model])
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(0, 80_000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    ax.set_title("S-label-flip over 5 seeds: mean (line) and min-max (band); every label inverts at 40k",
                 loc="left", fontsize=8.5)
    ax.set_ylabel("accuracy per 1k tweets", fontsize=7.5)
    ax.set_xlabel("instance index", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    style(ax)
    save(fig, "fig2_label_flip_seeds")


def grouped_barh(ax, values: pd.DataFrame, models, fmt, log=False, annotate=None) -> None:
    """values: index = streams (top to bottom), columns = models."""
    y = np.arange(len(values))
    h = 0.8 / len(models)
    for k, model in enumerate(models):
        pos = y - 0.4 + h / 2 + k * h
        vals = values[model].to_numpy(dtype=float)
        ax.barh(pos, np.where(np.isnan(vals), 0, vals), height=h * 0.9,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for yi, v in zip(pos, vals):
            text = annotate(v) if annotate else fmt(v)
            x = v if not np.isnan(v) and v > 0 else (1 if log else 0)
            ax.annotate(text, (x, yi), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=5.8, color=MUTED)
    ax.set_yticks(y, [STREAM_LABELS[s] for s in values.index], fontsize=7)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    style(ax, grid_axis="x")


def fig_detection(grid) -> None:
    models = ["baseline-Rebuild", "enhanced-or"]
    d = grid[grid.model.isin(models)]
    agg = d.groupby(["stream", "model"]).agg(delay=("model_mean_delay", "mean"),
                                             fa=("model_false_alarms", "mean"),
                                             missed=("model_missed", "sum"),
                                             drifts=("n_drifts", "sum"))
    streams = [s for s in STREAM_ORDER if s in agg.index.get_level_values(0)]
    delay = agg.delay.unstack().loc[streams, models]
    fa = agg.fa.unstack().loc[streams, models]
    miss = (agg.missed / agg.drifts).unstack().loc[streams, models]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.4), sharey=True, constrained_layout=True)
    grouped_barh(axes[0], delay, models, None, log=True,
                 annotate=lambda v: "none" if np.isnan(v) else f"{v:,.0f}")
    axes[0].set_xlim(10, 3e5)
    axes[0].set_title("(a) mean delay of the first\ndetection (instances, log)", loc="left", fontsize=8)
    grouped_barh(axes[1], fa, models, lambda v: f"{v:.1f}")
    axes[1].set_xlim(0, max(fa.max().max() * 1.35, 1))
    axes[1].set_title("(b) false alarms per run", loc="left", fontsize=8)
    grouped_barh(axes[2], miss, models, lambda v: f"{v:.0%}")
    axes[2].set_xlim(0, 1.18)
    axes[2].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[2].set_title("(c) share of drifts missed", loc="left", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in models]
    fig.legend(handles, [MODEL_LABELS[m] for m in models], loc="outside lower center", ncol=2,
               frameon=False, fontsize=7)
    save(fig, "fig3_detection")


def fig_recovery(grid) -> None:
    models = ["baseline-Rebuild", "enhanced-or"]
    d = grid[grid.model.isin(models)].copy()
    streams = ["s_label_flip", "s_recurring", "s_prior_shift", "s_sudden", "s_gradual",
               "s_label_flip_topic", "ds1", "ds2"]
    fig, ax = plt.subplots(figsize=(3.5, 3.3), constrained_layout=True)
    y = np.arange(len(streams))
    for k, model in enumerate(models):
        rows = d[d.model == model].set_index("stream")
        for yi, s in zip(y, streams):
            r = rows.loc[[s]]
            pos = yi - 0.2 + 0.4 * k
            if r.not_recovered.sum() == r.n_drifts.sum():
                ax.annotate("never recovers (NB ceiling)" if k == 0 else "", (1, pos),
                            fontsize=5.8, color=MUTED, va="center")
                continue
            vals = r.recovery_mean.dropna()
            mean = vals.mean()
            ax.barh(pos, max(mean, 1), height=0.36, color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model] if yi == 0 else None)
            right = mean
            if len(vals) > 1 and vals.max() > vals.min():
                ax.errorbar(max(mean, 1), pos, xerr=[[mean - vals.min()], [vals.max() - mean]],
                            color=INK, linewidth=0.6, capsize=1.5)
                right = vals.max()  # label after the whisker so the two never overlap
            ax.annotate(f"{mean:,.0f}", (max(right, 1), pos), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=5.8, color=MUTED)
    ax.set_yticks(y, [STREAM_LABELS[s] for s in streams], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 13_000)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    ax.set_xlabel("instances until accuracy is back to 95% of the\npre-drift level (0 = it never fell below)",
                  fontsize=7)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1,
               frameon=False, fontsize=6.3)
    style(ax, grid_axis="x")
    save(fig, "fig4_recovery")


def fig_ablation(grid) -> None:
    modes = ["enhanced-adwin_only", "enhanced-or", "enhanced-and"]
    acc = grid.pivot_table(index="stream", columns="model", values="accuracy", aggfunc="mean")
    gain = acc[modes].sub(acc["baseline-Rebuild"], axis=0).loc[STREAM_ORDER] * 100

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2), width_ratios=[1.5, 1],
                                  constrained_layout=True)
    grouped_barh(ax, gain, modes, lambda v: f"{v:+.2f}")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(min(gain.min().min() * 1.4, -0.5), gain.max().max() * 1.3)
    ax.set_xlabel("accuracy gain over baseline Rebuild-Zero (percentage points)", fontsize=7)
    ax.set_title("(a) accuracy gain per stream", loc="left", fontsize=8)

    d = grid[grid.model.isin(["baseline-Rebuild"] + modes)]
    tot = d.groupby("model").agg(detected=("model_detected", "sum"),
                                 false_alarms=("model_false_alarms", "sum"))
    tot = tot.loc[["baseline-Rebuild"] + modes].astype(int)
    y = np.arange(len(tot))
    ax2.barh(y - 0.18, tot.detected, height=0.34, color=[MODEL_COLORS[m] for m in tot.index])
    ax2.barh(y + 0.18, tot.false_alarms, height=0.34, color=[MODEL_COLORS[m] for m in tot.index],
             alpha=0.45, hatch="////", edgecolor="white", linewidth=0)
    for yi, (det, fa) in zip(y, tot.itertuples(index=False)):
        ax2.annotate(f"{det} detected", (det, yi - 0.18), xytext=(3, 0), textcoords="offset points",
                     va="center", fontsize=5.8, color=MUTED)
        ax2.annotate(f"{fa} false alarms", (fa, yi + 0.18), xytext=(3, 0), textcoords="offset points",
                     va="center", fontsize=5.8, color=MUTED)
    ax2.set_yticks(y, [MODEL_LABELS[m].replace("enhanced: ", "") for m in tot.index], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, tot.values.max() * 1.6)
    ax2.set_title(f"(b) over all {int(d[d.model == 'baseline-Rebuild'].n_drifts.sum())} drifts",
                  loc="left", fontsize=8)
    style(ax2, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in modes]
    fig.legend(handles, [MODEL_LABELS[m] for m in modes], loc="outside lower center", ncol=3,
               frameon=False, fontsize=7)
    save(fig, "fig5_ablation")


# -- tables -----------------------------------------------------------------------------

def table_accuracy(grid) -> None:
    def cell(x):
        return pct(x.mean()) if len(x) == 1 else f"{pct(x.mean())} ± {x.std(ddof=0) * 100:.1f}"
    t = grid.pivot_table(index="model", columns="stream", values="accuracy", aggfunc=cell)
    t = t.loc[MODEL_ORDER, STREAM_ORDER]
    t.to_csv(RESULTS / "table_accuracy.csv")
    out = t.reset_index()
    out["model"] = out["model"].map(MODEL_LABELS)
    out.columns = ["Model"] + [STREAM_LABELS[s] for s in STREAM_ORDER]
    write_tex(out, "table_accuracy", "Prequential accuracy (\\%) per stream; scenarios: mean "
              "$\\pm$ std over 5 seeds.", "tab:accuracy", "l" + "r" * len(STREAM_ORDER),
              wide=True)


def table_main(grid) -> pd.DataFrame:
    scen = grid[~grid.stream.isin(["ds1", "ds2"])]
    acc = grid.pivot_table(index="model", columns="stream", values="accuracy", aggfunc="mean")
    agg = grid.groupby("model").agg(
        precision=("precision_macro", "mean"), recall=("recall_macro", "mean"),
        f1=("f1_macro", "mean"), detected=("model_detected", "sum"),
        drifts=("n_drifts", "sum"), false_alarms=("model_false_alarms", "sum"),
        delay=("model_mean_delay", "mean"), recovery=("recovery_mean", "mean"),
        not_recovered=("not_recovered", "sum"),
        throughput=("throughput", "mean"), batch_p95=("batch_ms_p95", "mean"),
        entries=("model_entries", "mean"))
    agg["acc_scenarios"] = scen.groupby("model").accuracy.mean()
    agg["acc_ds1"], agg["acc_ds2"] = acc["ds1"], acc["ds2"]
    agg = agg.loc[MODEL_ORDER]
    agg.round(4).to_csv(RESULTS / "table_main.csv")

    adaptive = agg.index.str.startswith(("baseline", "enhanced"))
    out = pd.DataFrame({
        "Model": agg.index.map(MODEL_LABELS),
        "Acc. scen.": agg.acc_scenarios.map(pct), "Acc. DS1": agg.acc_ds1.map(pct),
        "Acc. DS2": agg.acc_ds2.map(pct), "P": agg.precision.map(pct), "R": agg.recall.map(pct),
        "F1": agg.f1.map(pct),
        "Detected": [f"{int(d)}/{int(n)}" if a else "-" for d, n, a in zip(agg.detected, agg.drifts, adaptive)],
        "FA": [f"{int(f)}" if a else "-" for f, a in zip(agg.false_alarms, adaptive)],
        "Delay": [f"{d:,.0f}" if a and not np.isnan(d) else "-" for d, a in zip(agg.delay, adaptive)],
        "Recovery": [f"{r:,.0f}" if not np.isnan(r) else "-" for r in agg.recovery],
        "Never rec.": agg.not_recovered.map(lambda v: f"{int(v)}"),
        "Tweets/s": agg.throughput.map(lambda v: f"{v / 1e3:.0f}k"),
        "Batch p95 (ms)": agg.batch_p95.map(lambda v: f"{v:.0f}"),
        "Entries": agg.entries.map(lambda v: f"{v / 1e3:.0f}k" if v else "-"),
    })
    write_tex(out, "table_main",
              "Main results over the grid (7 models x 8 streams, scenarios x 5 seeds). Acc., P, R, "
              "F1 in \\%; macro P/R/F1 averaged over all runs; detection over all 47 true drifts "
              "(FA = false alarms); delay and recovery in instances (recovery averages the drifts "
              "a model recovers from; Never rec. counts the others); cost from the offline runner "
              "(throughput, p95 time per 1k-tweet batch, (word, class) entries).",
              "tab:main", "l" + "r" * (len(out.columns) - 1), wide=True)
    return agg


def table_cost(grid) -> None:
    models = ["baseline-Rebuild", "enhanced-or"]
    d = grid[grid.model.isin(models)].groupby("model").agg(
        throughput=("throughput", "mean"), batch_p50=("batch_ms_p50", "mean"),
        batch_p95=("batch_ms_p95", "mean"), entries=("model_entries", "mean"),
        rss=("rss_growth_mb", "mean"))
    bench = pd.read_csv(RESULTS / "streaming_bench_s_label_flip.csv")
    b = bench[(bench.batch_cap == 1000) & (bench.target_rate == 1000)].iloc[0]
    d["stream_ms_per_1k"] = [b.baseline_ms_per_1k, b.enhanced_ms_per_1k]
    d["stream_accuracy"] = [b.baseline_accuracy, b.enhanced_accuracy]
    d.round(2).to_csv(RESULTS / "table_cost.csv")
    out = pd.DataFrame({
        "Model": d.index.map(MODEL_LABELS),
        "Offline tweets/s": d.throughput.map(lambda v: f"{v:,.0f}"),
        "Batch p50 / p95 (ms)": [f"{a:.0f} / {b:.0f}" for a, b in zip(d.batch_p50, d.batch_p95)],
        "Entries": d.entries.map(lambda v: f"{v:,.0f}"),
        "Spark ms per 1k": d.stream_ms_per_1k.map(lambda v: f"{v:.1f}"),
    })
    write_tex(out, "table_cost",
              "Cost of the enhancement. Offline: mean over the grid runs. Spark: model time per "
              "1k tweets in the streaming pipeline (S-label-flip, 1,000 tweets/s); the pipeline adds "
              "about 170 ms fixed overhead per micro-batch for both.",
              "tab:cost", "lrrrr")


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "hatch.color": "white"})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(RESULTS / "grid.csv")
    windows = pd.read_csv(RESULTS / "grid_windows.csv")
    events = pd.read_csv(RESULTS / "grid_events.csv")
    fig_accuracy_real(windows, events)
    fig_label_flip_seeds(windows)
    fig_detection(grid)
    fig_recovery(grid)
    fig_ablation(grid)
    table_accuracy(grid)
    agg = table_main(grid)
    table_cost(grid)
    with pd.option_context("display.width", 220):
        print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
