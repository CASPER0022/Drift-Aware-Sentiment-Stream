"""Spark Structured Streaming consumer: Kafka -> parse -> tokenize -> model -> InfluxDB.

Kafka reading, JSON parsing and tokenisation run on the executors. The model is
sequential and stateful (ageing MNB updates per tweet), so each micro-batch is collected
to the driver inside foreachBatch and evaluated prequentially there: predict, then learn,
one tweet at a time in Kafka offset order. The micro-batch size (maxOffsetsPerTrigger) is
the unit at which metrics are reported, and later the detector check window w.

One model instance is kept per producer `run`, so replays never share state.

Run on the Docker Spark cluster with:  bash pipeline/submit_consumer.sh [--batch-size 1000]
"""
import argparse
import functools
import time
from datetime import datetime, timezone

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, LongType, StringType, StructField, StructType

from core.majority import MajorityClass
from core.mnb import AccumulativeMNB, AgeingMNB
from core.preprocess import tokenize
from core.timeunit import TIME_UNITS, model_time
from influx_sink import InfluxSink

MODELS = {"majority": MajorityClass, "accumulative": AccumulativeMNB, "fading": AgeingMNB}

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


class BatchProcessor:
    """The foreachBatch callback; lives on the driver and owns the model state."""

    def __init__(self, model_name: str, make_model, sink: InfluxSink, time_unit: str) -> None:
        self.model_name = model_name
        self.make_model = make_model
        self.sink = sink
        self.time_unit = time_unit
        self.runs: dict[str, dict] = {}  # run -> {"model", "seen", "correct", "last_end"}

    def __call__(self, batch_df, batch_id: int) -> None:
        rows = batch_df.orderBy("offset").collect()
        if not rows:
            return
        t_start = time.perf_counter()
        by_run: dict[str, list] = {}
        for row in rows:
            by_run.setdefault(row.run, []).append(row)

        for run, run_rows in by_run.items():
            state = self.runs.setdefault(run, {"model": self.make_model(), "seen": 0,
                                               "correct": 0, "last_end": None})
            model = state["model"]
            correct = 0
            for row in run_rows:
                tokens = list(row.tokens)
                t = model_time(datetime.fromisoformat(row.ts), row.idx, self.time_unit)
                correct += int(model.predict_one(tokens, t) == row.label)
                model.learn_one(tokens, row.label, t)
            state["seen"] += len(run_rows)
            state["correct"] += correct

            now = time.time()
            now_ms = now * 1000
            latencies = np.array([now_ms - r.sent_ms for r in run_rows])
            processing_s = time.perf_counter() - t_start
            # Throughput = tweets handled per second of wall clock since this run's last batch.
            span = now - state["last_end"] if state["last_end"] else processing_s
            state["last_end"] = now

            tags = {"stream": run_rows[0].stream, "run": run, "model": self.model_name}
            self.sink.write_batch(tags, {
                "n": len(run_rows),
                "accuracy": correct / len(run_rows),
                "accuracy_cum": state["correct"] / state["seen"],
                "seen": state["seen"],
                "idx_last": run_rows[-1].idx,
                "pos_share": sum(r.label for r in run_rows) / len(run_rows),
                "latency_ms": latencies.mean(),
                "latency_p95_ms": np.percentile(latencies, 95),
                "processing_ms": processing_s * 1000,
                "throughput": len(run_rows) / max(span, 1e-6),
                "lambda": getattr(model, "lam", None),
            }, time=datetime.now(timezone.utc))
            for event in getattr(model, "drain_events", lambda: [])():
                self.sink.write_drift_event(tags, **event)

            print(f"batch {batch_id:>5} run {run}: n={len(run_rows):,} "
                  f"acc={correct / len(run_rows):.3f} cum={state['correct'] / state['seen']:.3f} "
                  f"seen={state['seen']:,} p95 latency={np.percentile(latencies, 95):,.0f} ms",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(MODELS), default="fading")
    ap.add_argument("--lam", type=float, default=0.2, help="ageing factor for fading")
    ap.add_argument("--time-unit", choices=TIME_UNITS, default="hour")
    ap.add_argument("--batch-size", type=int, default=1000, help="max tweets per micro-batch")
    ap.add_argument("--topic", default="tweets")
    ap.add_argument("--bootstrap", default="kafka:29092")
    ap.add_argument("--starting-offsets", default="latest", choices=["latest", "earliest"])
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint dir (default: a fresh one per start under /tmp)")
    args = ap.parse_args()

    spark = SparkSession.builder.appName(f"drift-stream-{args.model}").getOrCreate()
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

    # The model lives in driver memory, so a restart starts from scratch; a fresh checkpoint
    # keeps Kafka offsets consistent with that.
    checkpoint = args.checkpoint or f"/tmp/checkpoints/{args.model}-{int(time.time())}"
    make_model = (functools.partial(AgeingMNB, lam=args.lam) if args.model == "fading"
                  else MODELS[args.model])
    label = f"fading_lam{args.lam:g}" if args.model == "fading" else args.model  # Influx tag
    sink = InfluxSink.from_env()
    query = (tweets.writeStream
             .foreachBatch(BatchProcessor(label, make_model, sink, args.time_unit))
             .option("checkpointLocation", checkpoint)
             .start())
    print(f"consuming {args.topic!r} from {args.bootstrap} with model={args.model}, "
          f"batch<= {args.batch_size:,}, checkpoint {checkpoint}", flush=True)
    try:
        query.awaitTermination()
    finally:
        sink.close()


if __name__ == "__main__":
    main()
