"""Write per-batch stream metrics and drift events to InfluxDB 2.

Measurements:
  model_metrics     tags stream, run, model; one point per model per micro-batch:
                    accuracy, accuracy_cum, f1_macro, lambda, model_entries, processing_ms, seen
  pipeline_metrics  tags stream, run; one point per micro-batch for the whole pipeline:
                    n, throughput, latency_ms, latency_p95_ms, batch_ms, driver_rss_mb, idx_last
  drift_event       tags stream, run, model, detector (vocab | adwin), kind (warning | change
                    | rebuild); field idx (instance index within the run), text (annotation)

Connection settings come from INFLUX_URL / INFLUX_TOKEN / INFLUX_ORG / INFLUX_BUCKET; the
defaults match .env.example and the in-network address used by the Spark driver.
"""
import os
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxSink:
    def __init__(self, url: str, token: str, org: str, bucket: str) -> None:
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket, self.org = bucket, org
        self.pending: list[Point] = []

    @classmethod
    def from_env(cls) -> "InfluxSink":
        return cls(url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
                   token=os.getenv("INFLUX_TOKEN", "dev-token-group18"),
                   org=os.getenv("INFLUX_ORG", "group18"),
                   bucket=os.getenv("INFLUX_BUCKET", "drift"))

    def add(self, measurement: str, tags: dict, fields: dict, time: datetime | None = None) -> None:
        """Queue one point; numeric fields are stored as floats, None fields are skipped."""
        point = Point(measurement).time(time or datetime.now(timezone.utc))
        for key, value in tags.items():
            point = point.tag(key, value)
        for key, value in fields.items():
            if value is None:  # e.g. models without a decay rate have no lambda
                continue
            point = point.field(key, value if isinstance(value, str) else float(value))
        self.pending.append(point)

    def add_drift_event(self, tags: dict, idx: int, detector: str, kind: str,
                        time: datetime | None = None) -> None:
        self.add("drift_event", {**tags, "detector": detector, "kind": kind},
                 {"idx": idx, "text": f"{tags.get('model', '')}: {detector} {kind} at {idx:,}"},
                 time)

    def flush(self) -> None:
        """Send everything queued for this micro-batch in one request."""
        if self.pending:
            self.write_api.write(bucket=self.bucket, org=self.org, record=self.pending)
            self.pending = []

    def close(self) -> None:
        self.flush()
        self.client.close()
