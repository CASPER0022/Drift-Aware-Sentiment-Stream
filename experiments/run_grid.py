"""Run the full experiment grid (configs/grid.yaml): every model x every stream x seeds.

Each (stream, seed) is loaded once and every model is evaluated on it with
experiments/evaluate.py, so every row carries the same metric columns. DS1 / DS2 run one
at a time (each needs ~2 GB once tokenised); the 80k-tweet scenarios run in parallel.

Outputs (experiments/results/):
  grid.csv               one row per (stream, seed, model): quality, detection, recovery, cost
  grid_summary.csv       per (stream, model): mean and std over seeds of the main metrics
  grid_windows.csv       accuracy per eval window, for the accuracy-over-time figures
  grid_events.csv        detector changes and rebuilds (warnings omitted), for the figures
  grid_config.yaml       the fully resolved spec of every (stream, model), for reproducibility

Usage:  python experiments/run_grid.py [--workers 3] [--streams ds1 s_label_flip ...]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.factory import DEFAULTS, build_model  # noqa: E402
from experiments.evaluate import drift_meta, run_and_measure  # noqa: E402
from experiments.runner_offline import load_instances  # noqa: E402
from pipeline.streams import stream_path  # noqa: E402

CONFIG = ROOT / "experiments" / "configs" / "grid.yaml"
RESULTS = ROOT / "experiments" / "results"
LARGE = ("ds1", "ds2")
SUMMARY_METRICS = ["accuracy", "f1_macro", "accuracy_post_drift", "model_mean_delay",
                   "model_false_alarms", "model_missed", "recovery_mean", "throughput",
                   "batch_ms_p95", "model_entries"]


def resolved_specs(config: dict, stream: str) -> dict[str, dict]:
    s = config["streams"][stream]
    return {name: {**DEFAULTS, **spec, **s.get("params", {}), **s.get("overrides", {}).get(name, {})}
            for name, spec in config["models"].items()}


def run_stream(config: dict, stream: str, seed: int) -> tuple[list, list, list]:
    s = config["streams"][stream]
    eval_window = s.get("eval_window", config["defaults"]["eval_window"])
    path = stream_path(stream, seed)
    instances, _ = load_instances(path, config["defaults"]["time_unit"])
    meta = drift_meta(path)
    rows, windows, events = [], [], []
    for name, spec in resolved_specs(config, stream).items():
        model = build_model(spec)
        metrics, result = run_and_measure(model, instances, meta,
                                          tolerance=config["defaults"].get("tolerance"),
                                          recovery_window=config["defaults"].get("recovery_window"))
        rows.append({"stream": stream, "seed": seed, "model": name, "kind": spec["model"],
                     **metrics})
        ends, acc = result.windowed_accuracy(eval_window)
        windows += [{"stream": stream, "seed": seed, "model": name, "end_idx": int(e),
                     "accuracy": round(float(a), 5)} for e, a in zip(ends, acc)]
        events += [{"stream": stream, "seed": seed, "model": name, "idx": e["idx"],
                    "detector": e["detector"], "kind": e["kind"], "acted": e.get("acted")}
                   for e in getattr(model, "events", []) if e["kind"] != "warning"]
    return rows, windows, events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3, help="parallel scenario jobs")
    ap.add_argument("--streams", nargs="+", default=None, help="default: all streams in the config")
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text())
    streams = args.streams or list(config["streams"])
    jobs = [(st, seed) for st in streams for seed in config["streams"][st]["seeds"]]
    rows, windows, events = [], [], []
    t0 = time.perf_counter()

    def collect(result, stream, seed):
        rows.extend(result[0]); windows.extend(result[1]); events.extend(result[2])  # noqa: E702
        print(f"  done {stream} seed {seed} ({time.perf_counter() - t0:.0f}s, "
              f"{len(rows)} rows)", flush=True)

    for stream, seed in [j for j in jobs if j[0] in LARGE]:  # one at a time: memory
        collect(run_stream(config, stream, seed), stream, seed)
    small = [j for j in jobs if j[0] not in LARGE]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_stream, config, st, seed): (st, seed) for st, seed in small}
        for fut in as_completed(futures):
            collect(fut.result(), *futures[fut])

    grid = pd.DataFrame(rows).sort_values(["stream", "seed", "model"])
    expected = len(jobs) * len(config["models"])
    missing = expected - len(grid)
    assert missing == 0, f"{missing} grid cells missing"

    partial = args.streams is not None
    stem = "grid_partial" if partial else "grid"
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS / f"{stem}.csv", index=False)
    pd.DataFrame(windows).to_csv(RESULTS / f"{stem}_windows.csv", index=False)
    pd.DataFrame(events).to_csv(RESULTS / f"{stem}_events.csv", index=False)
    agg = grid.groupby(["stream", "model"], sort=False)[SUMMARY_METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "seeds", grid.groupby(["stream", "model"], sort=False).size())
    agg.reset_index().to_csv(RESULTS / f"{stem}_summary.csv", index=False)
    (RESULTS / f"{stem}_config.yaml").write_text(yaml.safe_dump(
        {st: resolved_specs(config, st) for st in streams}, sort_keys=False))

    table = grid.pivot_table(index="stream", columns="model", values="accuracy", aggfunc="mean")
    with pd.option_context("display.width", 220):
        print((table * 100).round(2).to_string())
    print(f"{len(grid)} / {expected} cells complete in {time.perf_counter() - t0:.0f}s; "
          f"wrote {RESULTS.relative_to(ROOT)}/{stem}*.csv")


if __name__ == "__main__":
    main()
