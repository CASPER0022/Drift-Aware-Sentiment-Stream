"""Write per-batch stream metrics and drift events to InfluxDB 2.

Measurements (tags: stream, run, model):
  batch_metrics  fields accuracy, accuracy_cum, n, latency_ms, latency_p95_ms,
                 processing_ms, throughput, lambda, idx_last, ...
  drift_event    fields idx; extra tags detector (vocab | adwin), kind (warning | change)

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

    @classmethod
    def from_env(cls) -> "InfluxSink":
        return cls(url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
                   token=os.getenv("INFLUX_TOKEN", "dev-token-group18"),
                   org=os.getenv("INFLUX_ORG", "group18"),
                   bucket=os.getenv("INFLUX_BUCKET", "drift"))

    def write_batch(self, tags: dict, fields: dict, time: datetime | None = None) -> None:
        point = Point("batch_metrics").time(time or datetime.now(timezone.utc))
        for key, value in tags.items():
            point = point.tag(key, value)
        for key, value in fields.items():
            if value is not None:  # e.g. models without a decay rate have no lambda
                point = point.field(key, float(value))
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def write_drift_event(self, tags: dict, idx: int, detector: str, kind: str,
                          time: datetime | None = None) -> None:
        point = (Point("drift_event").time(time or datetime.now(timezone.utc))
                 .tag("detector", detector).tag("kind", kind).field("idx", int(idx)))
        for key, value in tags.items():
            point = point.tag(key, value)
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def close(self) -> None:
        self.client.close()
