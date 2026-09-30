"""Run the baseline reproduction (configs/baseline.yaml) on DS1 and DS2.

Each stream is loaded and tokenised once, then every configured model is evaluated
prequentially on it. Outputs (experiments/results/):
  baseline_summary.csv   one row per (stream, config): accuracy, macro P/R/F1, accuracy
                         before/after the natural change point, detector counts, runtime
  baseline_windows.csv   accuracy per eval window (long format), for accuracy-over-time plots
  baseline_events.csv    detector warnings / changes / rebuilds of the informed models
  baseline_lambda.csv    lambda trace of the informed models

Usage:  python experiments/run_baseline.py [--streams ds1 ds2] [--only Rebuild-Init ...]
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.metrics import classification_summary, prequential  # noqa: E402
from experiments.runner_offline import load_instances, make_model  # noqa: E402
from pipeline.streams import stream_path  # noqa: E402

CONFIG = ROOT / "experiments" / "configs" / "baseline.yaml"
RESULTS = ROOT / "experiments" / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--streams", nargs="+", default=["ds1", "ds2"])
    ap.add_argument("--only", nargs="+", default=None, help="run only these config names")
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text())
    summaries, windows, events, lambdas = [], [], [], []
    for stream in args.streams:
        path = stream_path(stream)
        change_idx = json.loads(path.with_name(f"{path.stem}_drift_points.json").read_text())
        natural_cp = change_idx.get("natural_change_point", change_idx["drift_points"][-1])
        t0 = time.perf_counter()
        instances, df = load_instances(path, config["defaults"]["time_unit"])
        print(f"{stream}: loaded {len(df):,} tweets in {time.perf_counter() - t0:.0f}s, "
              f"natural change point {natural_cp:,}", flush=True)

        for name, cfg in config["streams"][stream].items():
            if args.only and name not in args.only:
                continue
            params = Namespace(**{**config["defaults"], "lam": 0.0, "strategy": None, "w": None,
                                  **cfg})
            model = make_model(params)
            t0 = time.perf_counter()
            result = prequential(model, instances)
            run_s = time.perf_counter() - t0

            y, p = result.y_true, result.y_pred
            pre, post = classification_summary(y[:natural_cp], p[:natural_cp]), \
                classification_summary(y[natural_cp:], p[natural_cp:])
            model_events = getattr(model, "events", [])
            summary = result.summary()
            summaries.append({
                "stream": stream, "config": name, "model": cfg["model"],
                "strategy": cfg.get("strategy", ""), "lam0": params.lam,
                "w": params.w or "", "accuracy": summary["accuracy"],
                "precision_macro": summary["precision_macro"],
                "recall_macro": summary["recall_macro"], "f1_macro": summary["f1_macro"],
                "accuracy_pre_change": pre["accuracy"], "accuracy_post_change": post["accuracy"],
                "n_changes": sum(e["kind"] == "change" for e in model_events),
                "n_warnings": sum(e["kind"] == "warning" for e in model_events),
                "first_change_after_cp": next((e["idx"] for e in model_events
                                               if e["kind"] == "change" and e["idx"] >= natural_cp), ""),
                "runtime_s": round(run_s, 1), "model_entries": model.size,
            })
            ends, acc = result.windowed_accuracy(params.eval_window)
            windows.append(pd.DataFrame({"stream": stream, "config": name, "end_idx": ends,
                                         "accuracy": acc.round(5)}))
            events += [{"stream": stream, "config": name, **e} for e in model_events]
            lambdas += [{"stream": stream, "config": name, "idx": i, "lambda": v}
                        for i, v in getattr(model, "lambda_trace", [])]
            print(f"  {name:30s} acc {summary['accuracy']:.4f}  pre {pre['accuracy']:.4f}  "
                  f"post {post['accuracy']:.4f}  changes {summaries[-1]['n_changes']:>2}  "
                  f"({run_s:.0f}s)", flush=True)
        del instances, df

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(summaries)
    num = out.select_dtypes(include=[np.floating]).columns
    out[num] = out[num].round(5)
    suffix = "" if args.only is None and args.streams == ["ds1", "ds2"] else "_partial"
    out.to_csv(RESULTS / f"baseline_summary{suffix}.csv", index=False)
    pd.concat(windows).to_csv(RESULTS / f"baseline_windows{suffix}.csv", index=False)
    pd.DataFrame(events).to_csv(RESULTS / f"baseline_events{suffix}.csv", index=False)
    pd.DataFrame(lambdas).to_csv(RESULTS / f"baseline_lambda{suffix}.csv", index=False)
    print(f"wrote {RESULTS.relative_to(ROOT)}/baseline_*{suffix}.csv")


if __name__ == "__main__":
    main()
