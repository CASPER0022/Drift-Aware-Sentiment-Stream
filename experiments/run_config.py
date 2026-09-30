"""Run every model of an experiment config (configs/*.yaml) on its streams.

Each stream is loaded and tokenised once, then every configured model is evaluated
prequentially on it. Outputs go to experiments/results/<config name>_*.csv:
  _summary.csv   one row per (stream, config): accuracy, macro P/R/F1, accuracy before and
                 after the main drift point, detector counts and first detection after it
  _windows.csv   accuracy per eval window (long format), for accuracy-over-time plots
  _events.csv    detector warnings / changes / rebuilds (vocab and ADWIN)
  _lambda.csv    lambda trace of the adaptive models

Config layout: `defaults` (model parameters), optional `stream_options` (per-stream
overrides such as eval_window or seed) and `streams` -> {config name: parameters}.

Usage:
  python experiments/run_config.py --config experiments/configs/baseline.yaml
  python experiments/run_config.py --config experiments/configs/enhancement.yaml --streams s_label_flip
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

RESULTS = ROOT / "experiments" / "results"
MODEL_DEFAULTS = {"lam": 0.0, "strategy": None, "w": None}


def first_after(events: list[dict], detector: str, point: int):
    return next((e["idx"] for e in events
                 if e["detector"] == detector and e["kind"] == "change" and e["idx"] >= point), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "baseline.yaml")
    ap.add_argument("--streams", nargs="+", default=None, help="default: every stream in the config")
    ap.add_argument("--only", nargs="+", default=None, help="run only these config names")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    prefix = args.config.stem
    streams = args.streams or list(config["streams"])
    summaries, windows, events, lambdas = [], [], [], []
    for stream in streams:
        options = {**config["defaults"], **config.get("stream_options", {}).get(stream, {})}
        path = stream_path(stream, options.get("seed", 0))
        meta = json.loads(path.with_name(f"{path.stem}_drift_points.json").read_text())
        drift_point = meta.get("natural_change_point", meta["drift_points"][-1])
        t0 = time.perf_counter()
        instances, df = load_instances(path, options["time_unit"])
        print(f"{path.stem}: loaded {len(df):,} tweets in {time.perf_counter() - t0:.0f}s, "
              f"main drift point {drift_point:,}", flush=True)

        for name, cfg in config["streams"][stream].items():
            if args.only and name not in args.only:
                continue
            params = Namespace(**{**options, **MODEL_DEFAULTS, **cfg})
            model = make_model(params)
            t0 = time.perf_counter()
            result = prequential(model, instances)
            run_s = time.perf_counter() - t0

            y, p = result.y_true, result.y_pred
            pre = classification_summary(y[:drift_point], p[:drift_point])
            post = classification_summary(y[drift_point:], p[drift_point:])
            model_events = getattr(model, "events", [])
            first_vocab = first_after(model_events, "vocab", drift_point)
            first_adwin = first_after(model_events, "adwin", drift_point)
            summary = result.summary()
            summaries.append({
                "stream": path.stem, "config": name, "model": cfg["model"],
                "strategy": cfg.get("strategy", ""), "fusion": cfg.get("fusion", ""),
                "lam0": params.lam, "w": params.w or "",
                "accuracy": summary["accuracy"], "precision_macro": summary["precision_macro"],
                "recall_macro": summary["recall_macro"], "f1_macro": summary["f1_macro"],
                "accuracy_pre_drift": pre["accuracy"], "accuracy_post_drift": post["accuracy"],
                "drift_point": drift_point,
                "vocab_changes": sum(e["detector"] == "vocab" and e["kind"] == "change"
                                     for e in model_events),
                "adwin_changes": sum(e["detector"] == "adwin" and e["kind"] == "change"
                                     for e in model_events),
                "vocab_warnings": sum(e["kind"] == "warning" for e in model_events),
                "rebuilds": sum(e["kind"] == "rebuild" for e in model_events),
                "vocab_delay": first_vocab - drift_point if first_vocab is not None else "",
                "adwin_delay": first_adwin - drift_point if first_adwin is not None else "",
                "runtime_s": round(run_s, 1), "model_entries": model.size,
            })
            ends, acc = result.windowed_accuracy(params.eval_window)
            windows.append(pd.DataFrame({"stream": path.stem, "config": name, "end_idx": ends,
                                         "accuracy": acc.round(5)}))
            events += [{"stream": path.stem, "config": name, **e} for e in model_events]
            lambdas += [{"stream": path.stem, "config": name, "idx": i, "lambda": v}
                        for i, v in getattr(model, "lambda_trace", [])]
            s = summaries[-1]
            print(f"  {name:30s} acc {summary['accuracy']:.4f}  pre {pre['accuracy']:.4f}  "
                  f"post {post['accuracy']:.4f}  delay vocab {s['vocab_delay']!s:>6} "
                  f"adwin {s['adwin_delay']!s:>6}  ({run_s:.0f}s)", flush=True)
        del instances, df

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(summaries)
    num = out.select_dtypes(include=[np.floating]).columns
    out[num] = out[num].round(5)
    partial = args.only is not None or set(streams) != set(config["streams"])
    stem = f"{prefix}{'_partial' if partial else ''}"
    out.to_csv(RESULTS / f"{stem}_summary.csv", index=False)
    pd.concat(windows).to_csv(RESULTS / f"{stem}_windows.csv", index=False)
    pd.DataFrame(events).to_csv(RESULTS / f"{stem}_events.csv", index=False)
    pd.DataFrame(lambdas).to_csv(RESULTS / f"{stem}_lambda.csv", index=False)
    print(f"wrote {RESULTS.relative_to(ROOT)}/{stem}_*.csv")


if __name__ == "__main__":
    main()
