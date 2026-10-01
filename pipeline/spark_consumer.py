"""Spark Structured Streaming consumer: Kafka -> parse -> tokenize -> models -> InfluxDB.

Kafka reading, JSON parsing and tokenisation run on the executors. The models are
sequential and stateful (ageing MNB updates per tweet), so each micro-batch is collected
to the driver inside foreachBatch and evaluated prequentially there: predict, then learn,
one tweet at a time in Kafka offset order.

Several models run side by side on the same batches (default: the baseline and the
enhanced model), so the dashboard compares them on identical data. Each producer `run`
gets fresh model instances, so replays never share state. Model specs are presets built
with core.factory; the vocabulary window w follows the stream as in the offline runs
(24k DS1, 22k DS2, 4k scenarios).

Run on the Docker Spark cluster with:
  bash pipeline/submit_consumer.sh                          # baseline + enhanced
  bash pipeline/submit_consumer.sh --models baseline enhanced accumulative --batch-size 1000
"""
import argparse
import os
import time
from datetime import datetime, timezone

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, LongType, StringType, StructField, StructType

from core.factory import build_model
from core.metrics import classification_summary
from core.preprocess import tokenize
from core.timeunit import TIME_UNITS, model_time
from influx_sink import InfluxSink

PRESETS = {  # the Day 8 frozen settings; see experiments/configs/enhancement.yaml
    "baseline": {"model": "informed", "strategy": "Rebuild", "lam": 0.0},
    "enhanced": {"model": "enhanced", "fusion": "or", "strategy": "Rebuild", "lam": 0.0},
    "enhanced_adwin_only": {"model": "enhanced", "fusion": "adwin_only", "strategy": "Rebuild", "lam": 0.0},
    "enhanced_and": {"model": "enhanced", "fusion": "and", "strategy": "Rebuild", "lam": 0.0},
    "accumulative": {"model": "accumulative"},
    "fading": {"model": "fading", "lam": 0.2},
    "majority": {"model": "majority"},
}
W_BY_STREAM = {"ds1": 24_000, "ds2": 22_000}  # everything else (scenarios): 4,000
KEEP_RUNS = 3  # model state of older runs is dropped so the driver's memory stays bounded

MESSAGE_SCHEMA = StructType([
    StructField("stream", StringType()),
    StructField("run", StringType()),
    StructField("idx", LongType()),
    StructField("id", LongType()),
    StructField("ts", StringType()),
    StructField("text", StringType()),
    StructField("label", IntegerType()),
    StructField("sent_ms", LongType()),
])


def driver_rss_mb() -> float | None:
    try:  # Linux only; the driver runs in the spark-master container
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return None


class BatchProcessor:
    """The foreachBatch callback; lives on the driver and owns all model state."""

    def __init__(self, model_names: list[str], sink: InfluxSink, time_unit: str) -> None:
        self.model_names = model_names
        self.sink = sink
        self.time_unit = time_unit
        self.runs: dict[str, dict] = {}

    def new_run(self, stream: str) -> dict:
        w = W_BY_STREAM.get(stream, 4_000)
        return {"models": {name: build_model({**PRESETS[name], "w": w}) for name in self.model_names},
                "seen": 0, "correct": {name: 0 for name in self.model_names}, "last_end": None}

    def __call__(self, batch_df, batch_id: int) -> None:
        t_batch = time.perf_counter()
        rows = batch_df.orderBy("offset").collect()
        if not rows:
            return
        by_run: dict[str, list] = {}
        for row in rows:
            by_run.setdefault(row.run, []).append(row)

        for run, run_rows in by_run.items():
            stream = run_rows[0].stream
            if run not in self.runs:
                self.runs[run] = self.new_run(stream)
                for old in list(self.runs)[:-KEEP_RUNS]:  # dicts keep insertion order
                    del self.runs[old]
            state = self.runs[run]
            instances = [(list(r.tokens), r.label,
                          model_time(datetime.fromisoformat(r.ts), r.idx, self.time_unit))
                         for r in run_rows]
            labels = np.array([r.label for r in run_rows])
            state["seen"] += len(run_rows)
            stamp = datetime.now(timezone.utc)
            summary_line = []

            for name, model in state["models"].items():
                t0 = time.perf_counter()
                preds = []
                for tokens, label, t in instances:
                    preds.append(model.predict_one(tokens, t))
                    model.learn_one(tokens, label, t)
                processing_ms = (time.perf_counter() - t0) * 1000
                preds = np.array(preds)
                summary = classification_summary(labels, preds)
                state["correct"][name] += int((preds == labels).sum())
                tags = {"stream": stream, "run": run, "model": name}
                self.sink.add("model_metrics", tags, {
                    "accuracy": summary["accuracy"],
                    "accuracy_cum": state["correct"][name] / state["seen"],
                    "f1_macro": summary["f1_macro"],
                    "lambda": getattr(model, "lam", None),
                    "model_entries": getattr(model, "size", None),
                    "processing_ms": processing_ms,
                    "seen": state["seen"],
                }, stamp)
                for event in getattr(model, "drain_events", lambda: [])():
                    if event["kind"] != "warning":  # warnings are frequent; keep the chart readable
                        self.sink.add_drift_event(tags, time=stamp, **event)
                summary_line.append(f"{name} {summary['accuracy']:.3f}")

            now = time.time()
            latencies = now * 1000 - np.array([r.sent_ms for r in run_rows])
            # Throughput: tweets per second of wall clock since this run's previous batch.
            span = now - state["last_end"] if state["last_end"] else time.perf_counter() - t_batch
            state["last_end"] = now
            self.sink.add("pipeline_metrics", {"stream": stream, "run": run}, {
                "n": len(run_rows),
                "throughput": len(run_rows) / max(span, 1e-6),
                "latency_ms": latencies.mean(),
                "latency_p95_ms": np.percentile(latencies, 95),
                "batch_ms": (time.perf_counter() - t_batch) * 1000,
                "driver_rss_mb": driver_rss_mb(),
                "idx_last": run_rows[-1].idx,
                "seen": state["seen"],
            }, stamp)
            self.sink.flush()
            print(f"batch {batch_id:>5} {run}: n={len(run_rows):,} seen={state['seen']:,} "
                  f"acc[{', '.join(summary_line)}] p95 latency={np.percentile(latencies, 95):,.0f} ms",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=list(PRESETS), default=["baseline", "enhanced"])
    ap.add_argument("--time-unit", choices=TIME_UNITS, default="hour")
    ap.add_argument("--batch-size", type=int, default=1000, help="max tweets per micro-batch")
    ap.add_argument("--topic", default="tweets")
    ap.add_argument("--bootstrap", default="kafka:29092")
    ap.add_argument("--starting-offsets", default="latest", choices=["latest", "earliest"])
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint dir (default: a fresh one per start under /tmp)")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("drift-stream").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    tokenize_udf = F.udf(tokenize, ArrayType(StringType()))
    tweets = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", args.bootstrap)
              .option("subscribe", args.topic)
              .option("startingOffsets", args.starting_offsets)
              .option("maxOffsetsPerTrigger", args.batch_size)
              .option("failOnDataLoss", "false")
              .load()
              .select(F.col("offset"),
                      F.from_json(F.col("value").cast("string"), MESSAGE_SCHEMA).alias("m"))
              .select("offset", "m.*")
              .withColumn("tokens", tokenize_udf("text"))
              .drop("text"))

    # The models live in driver memory, so a restart starts from scratch; a fresh checkpoint
    # keeps Kafka offsets consistent with that.
    checkpoint = args.checkpoint or f"/tmp/checkpoints/drift-stream-{int(time.time())}"
    sink = InfluxSink.from_env()
    query = (tweets.writeStream
             .foreachBatch(BatchProcessor(args.models, sink, args.time_unit))
             .option("checkpointLocation", checkpoint)
             .start())
    print(f"consuming {args.topic!r} from {args.bootstrap} with models {args.models}, "
          f"batch<= {args.batch_size:,}, checkpoint {checkpoint}", flush=True)
    try:
        query.awaitTermination()
    finally:
        sink.close()


if __name__ == "__main__":
    main()
