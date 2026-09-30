"""Offline runner: prequential evaluation of one model over one stream, no Kafka/Spark.

Uses the same core/ models, tokenizer and time unit as the streaming consumer, but
iterates the parquet file directly so the experiment grid runs fast.

Writes to experiments/results/:
  runs.csv                     one summary row per run (appended)
  windows/<run_id>.csv         accuracy per eval window, for accuracy-over-time plots
  events/<run_id>.json         informed models: detector checks, signals and lambda trace

Examples:
  python experiments/runner_offline.py --stream ds1 --model accumulative
  python experiments/runner_offline.py --stream ds1 --model fading --lam 0.2
  python experiments/runner_offline.py --stream s_label_flip --seed 0 --model fading --lam 0.2
  python experiments/runner_offline.py --stream ds1 --model informed --strategy FastSetFastReset       --lam 0.1 --lam-max 0.5 --w 24000
"""
import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adwin_detector import ErrorADWIN  # noqa: E402
from core.enhanced import ADWIN_REACTIONS, FUSION_MODES, EnhancedMNB  # noqa: E402
from core.informed import InformedAgeingMNB  # noqa: E402
from core.majority import MajorityClass  # noqa: E402
from core.metrics import prequential  # noqa: E402
from core.mnb import AccumulativeMNB, AgeingMNB  # noqa: E402
from core.preprocess import tokenize  # noqa: E402
from core.strategies import STRATEGIES, make_strategy  # noqa: E402
from core.timeunit import TIME_UNITS, model_time  # noqa: E402
from core.vocab_detector import VocabularyDetector  # noqa: E402
from pipeline.streams import stream_path  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def make_model(args):
    if args.model == "majority":
        return MajorityClass()
    if args.model == "accumulative":
        return AccumulativeMNB(alpha=args.alpha)
    if args.model == "fading":  # fadingMNB: ageing MNB with a fixed lambda (reference model)
        return AgeingMNB(lam=args.lam, alpha=args.alpha)
    if args.model in ("informed", "enhanced"):
        detector = VocabularyDetector(w=args.w, alpha=args.detect_alpha, beta=args.detect_beta,
                                      history=args.detect_history,
                                      reference=args.detect_reference)
        strategy = make_strategy(args.strategy, lam0=args.lam, lam_max=args.lam_max, c=args.c,
                                 decrease=args.decrease)
        if args.model == "informed":  # the baseline: vocabulary detector + lambda strategy
            return InformedAgeingMNB(detector, strategy, alpha=args.alpha)
        return EnhancedMNB(detector, strategy, ErrorADWIN(delta=args.adwin_delta),
                           fusion=args.fusion, adwin_reaction=args.adwin_reaction,
                           confirm_window=args.confirm_window, cooldown=args.cooldown,
                           min_rebuild=args.min_rebuild, lam_max=args.lam_max, alpha=args.alpha)
    raise ValueError(args.model)


def model_label(args) -> str:
    if args.model == "fading":
        return f"fading_lam{args.lam:g}"
    if args.model == "informed":
        return f"informed_{args.strategy}_lam{args.lam:g}_w{args.w}_h{args.detect_history}"
    if args.model == "enhanced":
        return (f"enhanced_{args.fusion}_{args.adwin_reaction}_d{args.adwin_delta:g}_"
                f"{args.strategy}_lam{args.lam:g}_w{args.w}")
    return args.model


def load_instances(path: Path, time_unit: str) -> tuple[list, pd.DataFrame]:
    df = pd.read_parquet(path, columns=["idx", "ts", "label", "text"])
    tokens = [tokenize(t) for t in df["text"]]
    times = [model_time(ts, i, time_unit) for ts, i in zip(df["ts"].dt.to_pydatetime(), df["idx"])]
    return list(zip(tokens, df["label"].tolist(), times)), df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", default="ds1")
    ap.add_argument("--seed", type=int, default=0, help="scenario seed")
    ap.add_argument("--model", choices=["majority", "accumulative", "fading", "informed",
                                        "enhanced"], default="fading")
    ap.add_argument("--lam", type=float, default=0.2,
                    help="ageing factor for fading; initial lam0 for informed")
    ap.add_argument("--strategy", choices=STRATEGIES, default="FastSetFastReset")
    ap.add_argument("--lam-max", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1, help="lambda step of the SlowIncrease strategies")
    ap.add_argument("--decrease", type=float, default=0.05, help="FastSetSlowDecrease step")
    ap.add_argument("--w", type=int, default=24_000, help="detector check window (instances)")
    ap.add_argument("--detect-alpha", type=float, default=1.8, help="change threshold (sigmas)")
    ap.add_argument("--detect-beta", type=float, default=0.334, help="warning threshold (sigmas)")
    ap.add_argument("--detect-history", type=int, default=20,
                    help="checks in the detector's moving mean/std")
    ap.add_argument("--detect-reference", choices=["previous", "accumulated"], default="previous")
    ap.add_argument("--fusion", choices=FUSION_MODES, default="or", help="enhanced: signal fusion")
    ap.add_argument("--adwin-delta", type=float, default=0.002, help="enhanced: ADWIN confidence")
    ap.add_argument("--adwin-reaction", choices=ADWIN_REACTIONS, default="rebuild")
    ap.add_argument("--confirm-window", type=int, default=None,
                    help="enhanced 'and' fusion: max distance between the two signals (default w)")
    ap.add_argument("--cooldown", type=int, default=2000, help="enhanced: instances after a reaction")
    ap.add_argument("--min-rebuild", type=int, default=1000, help="enhanced: min rebuild size")
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

    model = make_model(args)
    if args.memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    result = prequential(model, instances)
    run_s = time.perf_counter() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20 if args.memory else None
    if args.memory:
        tracemalloc.stop()

    label = model_label(args)
    run_id = f"{path.stem}__{label}__{args.time_unit}__{datetime.now():%Y%m%d-%H%M%S}"
    summary = result.summary()
    row = {"run_id": run_id, "stream": path.stem, "model": args.model,
           "lam": args.lam if args.model in ("fading", "informed") else "",
           "strategy": args.strategy if args.model in ("informed", "enhanced") else "",
           "w": args.w if args.model in ("informed", "enhanced") else "",
           "detect_history": args.detect_history if args.model in ("informed", "enhanced") else "",
           "detect_reference": args.detect_reference if args.model in ("informed", "enhanced") else "",
           "fusion": args.fusion if args.model == "enhanced" else "",
           "adwin_delta": args.adwin_delta if args.model == "enhanced" else "",
           "adwin_reaction": args.adwin_reaction if args.model == "enhanced" else "",
           "alpha": args.alpha,
           "time_unit": args.time_unit, "eval_window": eval_window,
           **{k: round(v, 5) if isinstance(v, float) else v for k, v in summary.items()},
           "runtime_s": round(run_s, 1), "throughput": round(len(df) / run_s),
           "model_entries": getattr(model, "size", ""),
           "n_changes": sum(e["kind"] == "change" for e in getattr(model, "events", [])),
           "n_warnings": sum(e["kind"] == "warning" for e in getattr(model, "events", [])),
           "peak_mem_mb": round(peak_mb, 1) if peak_mb is not None else ""}

    (RESULTS / "windows").mkdir(parents=True, exist_ok=True)
    ends, acc = result.windowed_accuracy(eval_window)
    pd.DataFrame({"end_idx": ends, "accuracy": acc.round(5)}).to_csv(
        RESULTS / "windows" / f"{run_id}.csv", index=False)
    if args.model in ("informed", "enhanced"):
        (RESULTS / "events").mkdir(parents=True, exist_ok=True)
        (RESULTS / "events" / f"{run_id}.json").write_text(json.dumps(
            {"events": model.events, "lambda_trace": model.lambda_trace, "checks": model.checks},
            indent=1))
    runs_csv = RESULTS / "runs.csv"
    runs = pd.DataFrame([row])
    if runs_csv.exists():  # concat keeps older rows readable when new columns appear
        runs = pd.concat([pd.read_csv(runs_csv), runs], ignore_index=True)
    runs.to_csv(runs_csv, index=False)

    print(f"{path.stem} | {label} | t in {args.time_unit}s | {len(df):,} tweets "
          f"(load {load_s:.0f}s, run {run_s:.0f}s = {len(df) / run_s:,.0f}/s)")
    print(f"  accuracy {summary['accuracy']:.4f}   macro P {summary['precision_macro']:.4f}  "
          f"R {summary['recall_macro']:.4f}  F1 {summary['f1_macro']:.4f}"
          + (f"   entries {row['model_entries']:,}" if row["model_entries"] != "" else "")
          + (f"   peak {peak_mb:.0f} MB" if peak_mb else ""))
    if args.model in ("informed", "enhanced"):
        for det in ("vocab", "adwin"):
            changes = [e["idx"] for e in model.events if e["detector"] == det and e["kind"] == "change"]
            rebuilds = [e["idx"] for e in model.events if e["detector"] == det and e["kind"] == "rebuild"]
            if changes or det == "vocab":
                print(f"  {det}: {len(changes)} changes at {changes}"
                      + (f"; rebuilds at {rebuilds}" if rebuilds else ""))
        print(f"  lambda trace: {[(i, round(v, 3)) for i, v in model.lambda_trace]}")
    print(f"  saved {run_id}")


if __name__ == "__main__":
    main()
