"""The model clock `t` used by the ageing models, shared by the offline and streaming runners.

  hour      hours since the Unix epoch, from the tweet timestamp (default). This is the
            unit that makes the paper's lambda values sensible: with lambda = 0.2 per
            hour a class unseen for a day keeps e^-4.8 ~ 1% of its weight.
  instance  the 0-based instance index. With lambda = 0.2 per instance a word unseen for
            100 tweets would keep e^-20 ~ 2e-9, so it needs much smaller lambdas.
"""
from datetime import datetime, timezone

TIME_UNITS = ("hour", "instance")
_EPOCH = datetime(1970, 1, 1)


def model_time(ts: datetime, idx: int, unit: str = "hour") -> float:
    if unit == "hour":
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return (ts - _EPOCH).total_seconds() / 3600.0
    if unit == "instance":
        return float(idx)
    raise ValueError(f"unknown time unit {unit!r}; expected one of {TIME_UNITS}")
