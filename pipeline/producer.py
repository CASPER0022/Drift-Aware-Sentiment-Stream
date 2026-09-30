"""Replay a stream (DS1, DS2 or a scenario) into Kafka in arrival order.

Each message is JSON:
  {stream, run, idx, id, ts, text, label, sent_ms}
`run` identifies one replay so the consumer keeps a separate model per replay, and
`sent_ms` (wall clock at send time) lets the consumer measure end-to-end latency.

Examples:
  python pipeline/producer.py --stream ds1 --limit 50000 --rate 2000
  python pipeline/producer.py --stream s_label_flip --seed 0 --rate 500
"""
import argparse
import json
import time
from datetime import datetime

import pandas as pd
from confluent_kafka import KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from streams import PROCESSED, SCENARIOS


def stream_path(name: str, seed: int):
    if name in ("ds1", "ds2"):
        return PROCESSED / f"{name}.parquet"
    return SCENARIOS / f"{name}_seed{seed}.parquet"


def ensure_topic(bootstrap: str, topic: str) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic in admin.list_topics(timeout=10).topics:
        return
    # One partition: Kafka only guarantees order within a partition, and the models
    # must see tweets in arrival order.
    future = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic]
    try:
        future.result()
        print(f"created topic {topic!r}")
    except KafkaException as exc:  # created concurrently by someone else
        if "TOPIC_ALREADY_EXISTS" not in str(exc):
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", default="ds1",
                    help="ds1, ds2 or a scenario name such as s_label_flip")
    ap.add_argument("--seed", type=int, default=0, help="scenario seed")
    ap.add_argument("--start", type=int, default=0, help="first instance index to send")
    ap.add_argument("--limit", type=int, default=None, help="number of tweets to send")
    ap.add_argument("--rate", type=float, default=1000, help="tweets per second (0 = unthrottled)")
    ap.add_argument("--topic", default="tweets")
    ap.add_argument("--bootstrap", default="localhost:9092")
    args = ap.parse_args()

    path = stream_path(args.stream, args.seed)
    df = pd.read_parquet(path, columns=["idx", "id", "ts", "text", "label"])
    df = df.iloc[args.start: None if args.limit is None else args.start + args.limit]
    stream = path.stem
    run = f"{stream}-{datetime.now():%Y%m%d-%H%M%S}"

    ensure_topic(args.bootstrap, args.topic)
    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 20,
                         "enable.idempotence": True})  # no duplicates or reordering on retry
    print(f"sending {len(df):,} tweets from {stream} as run {run} "
          f"at {'max' if args.rate <= 0 else f'{args.rate:g}/s'} to {args.topic!r}")

    t0 = time.perf_counter()
    for i, row in enumerate(df.itertuples(index=False)):
        if args.rate > 0:  # pace against the schedule, not per message, so errors don't accumulate
            ahead = t0 + i / args.rate - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
        msg = {"stream": stream, "run": run, "idx": int(row.idx), "id": int(row.id),
               "ts": row.ts.isoformat(), "text": row.text, "label": int(row.label),
               "sent_ms": int(time.time() * 1000)}
        while True:
            try:
                producer.produce(args.topic, json.dumps(msg).encode("utf-8"))
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.1)
        producer.poll(0)
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:>9,} sent ({(i + 1) / (time.perf_counter() - t0):,.0f}/s)")

    producer.flush()
    elapsed = time.perf_counter() - t0
    print(f"done: {len(df):,} tweets in {elapsed:.1f}s ({len(df) / elapsed:,.0f}/s), run {run}")


if __name__ == "__main__":
    main()
