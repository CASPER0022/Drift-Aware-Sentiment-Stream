"""Streaming benchmark: replay a stream through Kafka + Spark at several producer rates.

Needs the stack up and the consumer running (bash pipeline/submit_consumer.sh). For each
rate the producer replays the stream, the script waits until the consumer has processed
every tweet of that run, then reads the run's metrics back from InfluxDB:

  achieved throughput   tweets / (last batch time - first batch time)
  latency               producer send -> models done: median of per-batch means, p95 of
                        per-batch p95s, and the worst batch
  batch time            p50 / p95 of the foreachBatch duration (all models + Influx write)
  model cost            mean ms per 1k tweets for each model (shows what ADWIN adds)
  driver memory         peak RSS of the Spark driver

Writes experiments/results/streaming_bench.csv.
Usage:  python experiments/streaming_bench.py [--stream s_label_flip] [--rates 500 1000 2000 4000]
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def influx() -> InfluxDBClient:
    return InfluxDBClient(url=os.getenv("INFLUX_HOST_URL", "http://localhost:8086"),
                          token=os.getenv("INFLUX_TOKEN", "dev-token-group18"),
                          org=os.getenv("INFLUX_ORG", "group18"))


def fetch(client, measurement: str, run: str) -> pd.DataFrame:
    flux = (f'from(bucket: "drift") |> range(start: -1d) '
            f'|> filter(fn: (r) => r._measurement == "{measurement}" and r.run == "{run}") '
            f'|> pivot(rowKey: ["_time", "model"], columnKey: ["_field"], valueColumn: "_value")'
            if measurement == "model_metrics" else
            f'from(bucket: "drift") |> range(start: -1d) '
            f'|> filter(fn: (r) => r._measurement == "{measurement}" and r.run == "{run}") '
            f'|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")')
    df = client.query_api().query_data_frame(flux)
    if isinstance(df, list):
        df = pd.concat(df) if df else pd.DataFrame()
    return df


def replay(stream: str, rate: float, start: int, limit: int | None) -> tuple[str, int]:
    cmd = [sys.executable, str(ROOT / "pipeline" / "producer.py"), "--stream", stream,
           "--rate", str(rate), "--start", str(start)]
    if limit:
        cmd += ["--limit", str(limit)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    run = re.search(r"as run (\S+)", out).group(1)
    n = int(re.search(r"done: ([\d,]+) tweets", out).group(1).replace(",", ""))
    return run, n


def wait_until_done(client, run: str, n: int, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        df = fetch(client, "pipeline_metrics", run)
        if len(df) and df["seen"].max() >= n:
            return
        time.sleep(5)
    raise TimeoutError(f"consumer did not finish run {run} within {timeout}s")


def summarise(client, run: str, rate: float, n: int) -> dict:
    pipe = fetch(client, "pipeline_metrics", run).sort_values("_time")
    models = fetch(client, "model_metrics", run)
    span = (pipe["_time"].iloc[-1] - pipe["_time"].iloc[0]).total_seconds()
    row = {"run": run, "target_rate": rate, "n": n, "batches": len(pipe),
           "mean_batch_size": pipe["n"].mean(),
           "achieved_throughput": n / span if span > 0 else None,
           "latency_median_ms": pipe["latency_ms"].median(),
           "latency_p95_ms": np.percentile(pipe["latency_p95_ms"], 95),
           "latency_max_ms": pipe["latency_p95_ms"].max(),
           "batch_ms_p50": pipe["batch_ms"].median(),
           "batch_ms_p95": np.percentile(pipe["batch_ms"], 95),
           "driver_rss_peak_mb": pipe["driver_rss_mb"].max()}
    seen = pipe.set_index("_time")["n"]
    for name, rows in models.groupby("model"):
        per_batch = rows.set_index("_time")["processing_ms"]
        row[f"{name}_ms_per_1k"] = 1000 * per_batch.sum() / seen.reindex(per_batch.index).sum()
        row[f"{name}_accuracy"] = rows.sort_values("_time")["accuracy_cum"].iloc[-1]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", default="s_label_flip")
    ap.add_argument("--rates", type=float, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = influx()
    rows = []
    for rate in args.rates:
        print(f"replaying {args.stream} at {rate:g}/s ...", flush=True)
        run, n = replay(args.stream, rate, args.start, args.limit)
        wait_until_done(client, run, n)
        time.sleep(3)  # let the last points land
        rows.append(summarise(client, run, rate, n))
        r = rows[-1]
        print(f"  {run}: {r['achieved_throughput']:,.0f} tweets/s, latency median "
              f"{r['latency_median_ms']:,.0f} ms / p95 {r['latency_p95_ms']:,.0f} ms, "
              f"batch p50 {r['batch_ms_p50']:,.0f} ms", flush=True)

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"streaming_bench_{args.stream}.csv"
    out.round(2).to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}")
    client.close()


if __name__ == "__main__":
    main()
