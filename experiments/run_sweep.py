"""Sensitivity sweep on the controlled scenarios: ADWIN delta x vocabulary alpha x fusion.

Every model shares the base of Day 7 (Rebuild strategy, lambda0 = 0, w = 4,000); rows
differ only in the detector settings:
  baseline (vocabulary only)  detect_alpha in ALPHAS
  enhanced adwin_only         adwin_delta in DELTAS
  enhanced or / and           every (adwin_delta, detect_alpha) pair
Streams: every scenario x SEEDS. Outputs (experiments/results/):
  sweep_rows.csv     one full metrics row per (scenario, seed, model)
  sweep_summary.csv  per model setting: mean accuracy / delay / recovery and summed false
                     alarms and misses over all scenarios and seeds

Usage:  python experiments/run_sweep.py [--workers 4] [--seeds 0 1 2]
"""
import argparse
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate import drift_meta, run_and_measure  # noqa: E402
from experiments.runner_offline import load_instances, make_model  # noqa: E402
from pipeline.streams import stream_path  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
SCENARIOS = ["s_sudden", "s_gradual", "s_recurring", "s_label_flip", "s_prior_shift",
             "s_label_flip_topic"]
DELTAS = [0.002, 0.01, 0.05]
ALPHAS = [1.5, 1.8, 2.5]
BASE = dict(alpha=1.0, time_unit="hour", detect_beta=0.334, detect_history=20,
            detect_reference="previous", lam_max=0.5, c=0.1, decrease=0.05,
            adwin_reaction="rebuild", confirm_window=None, cooldown=2000, min_rebuild=100,
            strategy="Rebuild", lam=0.0, w=4000, detect_alpha=1.8, adwin_delta=0.002)


def settings() -> list[dict]:
    out = [{"setting": f"baseline a={a}", "model": "informed", "detect_alpha": a} for a in ALPHAS]
    out += [{"setting": f"adwin_only d={d}", "model": "enhanced", "fusion": "adwin_only",
             "adwin_delta": d} for d in DELTAS]
    out += [{"setting": f"{f} d={d} a={a}", "model": "enhanced", "fusion": f,
             "adwin_delta": d, "detect_alpha": a} for f in ("or", "and")
            for d, a in product(DELTAS, ALPHAS)]
    return out


def run_stream(scenario: str, seed: int) -> list[dict]:
    path = stream_path(scenario, seed)
    instances, _ = load_instances(path, BASE["time_unit"])
    meta = drift_meta(path)
    rows = []
    for s in settings():
        params = Namespace(**{**BASE, **s})
        metrics, _ = run_and_measure(make_model(params), instances, meta)
        rows.append({"scenario": scenario, "seed": seed, **s, **metrics})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    jobs = list(product(SCENARIOS, args.seeds))
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_stream, sc, seed): (sc, seed) for sc, seed in jobs}
        for done, fut in enumerate(as_completed(futures), 1):
            rows += fut.result()
            sc, seed = futures[fut]
            print(f"[{done}/{len(jobs)}] {sc} seed {seed} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["scenario", "seed", "setting"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "sweep_rows.csv", index=False)
    summary = df.groupby("setting", sort=False).agg(
        model=("model", "first"), fusion=("fusion", "first"),
        adwin_delta=("adwin_delta", "first"), detect_alpha=("detect_alpha", "first"),
        accuracy=("accuracy", "mean"), f1_macro=("f1_macro", "mean"),
        mean_delay=("model_mean_delay", "mean"), false_alarms=("model_false_alarms", "sum"),
        missed=("model_missed", "sum"), recovery_mean=("recovery_mean", "mean"),
        not_recovered=("not_recovered", "sum"), runs=("accuracy", "size"),
    ).reset_index().sort_values("accuracy", ascending=False)
    summary.to_csv(RESULTS / "sweep_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))
    print(f"wrote {RESULTS.relative_to(ROOT)}/sweep_rows.csv and sweep_summary.csv")


if __name__ == "__main__":
    main()
