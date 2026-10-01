"""Run every model of an experiment config (configs/*.yaml) on its streams.

Each stream is loaded and tokenised once, then every configured model is evaluated
prequentially on it. Outputs go to experiments/results/<config name>_*.csv:
  _summary.csv   one row per (stream, config) with every metric of experiments/evaluate.py:
                 quality, drift detection (delay / misses / false alarms), recovery, system cost
  _windows.csv   accuracy per eval window (long format), for accuracy-over-time plots
  _events.csv    detector warnings / changes / rebuilds (vocab and ADWIN)
  _lambda.csv    lambda trace of the adaptive models

Config layout: `defaults` (model parameters), optional `stream_options` (per-stream
overrides such as eval_window, seed, tolerance, recovery_window) and
`streams` -> {config name: parameters}.

Usage:
  python experiments/run_config.py --config experiments/configs/baseline.yaml
  python experiments/run_config.py --config experiments/configs/enhancement.yaml --streams s_label_flip
"""
import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate import drift_meta, run_and_measure  # noqa: E402
from experiments.runner_offline import load_instances, make_model  # noqa: E402
from pipeline.streams import stream_path  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
MODEL_DEFAULTS = {"lam": 0.0, "strategy": None, "w": None}


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
        meta = drift_meta(path)
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
            metrics, result = run_and_measure(model, instances, meta,
                                              tolerance=options.get("tolerance"),
                                              recovery_window=options.get("recovery_window"))
            adaptive = cfg["model"] in ("informed", "enhanced")
            summaries.append({
                "stream": path.stem, "config": name, "model": cfg["model"],
                "strategy": cfg.get("strategy"), "fusion": cfg.get("fusion"),
                "lam0": params.lam, "w": params.w,
                "detect_alpha": params.detect_alpha if adaptive else None,
                "adwin_delta": params.adwin_delta if cfg["model"] == "enhanced" else None,
                **metrics})
            ends, acc = result.windowed_accuracy(params.eval_window)
            windows.append(pd.DataFrame({"stream": path.stem, "config": name, "end_idx": ends,
                                         "accuracy": acc.round(5)}))
            model_events = getattr(model, "events", [])
            events += [{"stream": path.stem, "config": name, **e} for e in model_events]
            lambdas += [{"stream": path.stem, "config": name, "idx": i, "lambda": v}
                        for i, v in getattr(model, "lambda_trace", [])]
            m = metrics
            delay = lambda k: "-" if m.get(k) is None else f"{m[k]:,.0f}"  # noqa: E731
            print(f"  {name:30s} acc {m['accuracy']:.4f}  delay model {delay('model_mean_delay'):>6}"
                  f"  FA {m.get('model_false_alarms', '-')!s:>2}  missed {m.get('model_missed', '-')!s:>2}"
                  f"  recovery {delay('recovery_mean'):>6}  ({m['runtime_s']:.0f}s)", flush=True)
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
