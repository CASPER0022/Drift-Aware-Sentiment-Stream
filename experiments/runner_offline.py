"""Offline runner: prequential evaluation of one model over one stream, no Kafka/Spark.

Uses the same core/ models, tokenizer and time unit as the streaming consumer, but
iterates the parquet file directly so the experiment grid runs fast.

Writes to experiments/results/:
  runs.csv                     one summary row per run (appended)
  windows/<run_id>.csv         accuracy per eval window, for accuracy-over-time plots

Examples:
  python experiments/runner_offline.py --stream ds1 --model accumulative
  python experiments/runner_offline.py --stream ds1 --model fading --lam 0.2
  python experiments/runner_offline.py --stream s_label_flip --seed 0 --model fading --lam 0.2
"""
import argparse
import csv
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.majority import MajorityClass  # noqa: E402
from core.metrics import prequential  # noqa: E402
from core.mnb import AccumulativeMNB, AgeingMNB  # noqa: E402
from core.preprocess import tokenize  # noqa: E402
from core.timeunit import TIME_UNITS, model_time  # noqa: E402
from pipeline.streams import stream_path  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def make_model(name: str, lam: float, alpha: float):
    if name == "majority":
        return MajorityClass()
    if name == "accumulative":
        return AccumulativeMNB(alpha=alpha)
    if name == "fading":  # fadingMNB: ageing MNB with a fixed lambda (reference model)
        return AgeingMNB(lam=lam, alpha=alpha)
    raise ValueError(name)


def load_instances(path: Path, time_unit: str) -> tuple[list, pd.DataFrame]:
    df = pd.read_parquet(path, columns=["idx", "ts", "label", "text"])
    tokens = [tokenize(t) for t in df["text"]]
    times = [model_time(ts, i, time_unit) for ts, i in zip(df["ts"].dt.to_pydatetime(), df["idx"])]
    return list(zip(tokens, df["label"].tolist(), times)), df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", default="ds1")
    ap.add_argument("--seed", type=int, default=0, help="scenario seed")
    ap.add_argument("--model", choices=["majority", "accumulative", "fading"], default="fading")
    ap.add_argument("--lam", type=float, default=0.2, help="ageing factor for fading")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    ap.add_argument("--time-unit", choices=TIME_UNITS, default="hour")
    ap.add_argument("--eval-window", type=int, default=None,
                    help="instances per accuracy window (default: 10k for DS1/DS2, 1k otherwise)")
    ap.add_argument("--memory", action="store_true",
                    help="track peak Python memory with tracemalloc (slows the run ~2x)")
    args = ap.parse_args()

    path = stream_path(args.stream, args.seed)
    t0 = time.perf_counter()
    instances, df = load_instances(path, args.time_unit)
    load_s = time.perf_counter() - t0
    eval_window = args.eval_window or (10_000 if len(df) > 500_000 else 1_000)

    model = make_model(args.model, args.lam, args.alpha)
    if args.memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    result = prequential(model, instances)
    run_s = time.perf_counter() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20 if args.memory else None
    if args.memory:
        tracemalloc.stop()

    label = args.model if args.model != "fading" else f"fading_lam{args.lam:g}"
    run_id = f"{path.stem}__{label}__{args.time_unit}__{datetime.now():%Y%m%d-%H%M%S}"
    summary = result.summary()
    row = {"run_id": run_id, "stream": path.stem, "model": args.model,
           "lam": model.lam if hasattr(model, "lam") else "", "alpha": args.alpha,
           "time_unit": args.time_unit, "eval_window": eval_window,
           **{k: round(v, 5) if isinstance(v, float) else v for k, v in summary.items()},
           "runtime_s": round(run_s, 1), "throughput": round(len(df) / run_s),
           "model_entries": getattr(model, "size", ""),
           "peak_mem_mb": round(peak_mb, 1) if peak_mb is not None else ""}

    (RESULTS / "windows").mkdir(parents=True, exist_ok=True)
    ends, acc = result.windowed_accuracy(eval_window)
    pd.DataFrame({"end_idx": ends, "accuracy": acc.round(5)}).to_csv(
        RESULTS / "windows" / f"{run_id}.csv", index=False)
    runs_csv = RESULTS / "runs.csv"
    new_file = not runs_csv.exists()
    with runs_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print(f"{path.stem} | {label} | t in {args.time_unit}s | {len(df):,} tweets "
          f"(load {load_s:.0f}s, run {run_s:.0f}s = {len(df) / run_s:,.0f}/s)")
    print(f"  accuracy {summary['accuracy']:.4f}   macro P {summary['precision_macro']:.4f}  "
          f"R {summary['recall_macro']:.4f}  F1 {summary['f1_macro']:.4f}"
          + (f"   entries {row['model_entries']:,}" if row["model_entries"] != "" else "")
          + (f"   peak {peak_mb:.0f} MB" if peak_mb else ""))
    print(f"  saved {run_id}")


if __name__ == "__main__":
    main()
