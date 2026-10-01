"""Run one model over one stream and measure everything (Day 8 metrics).

`run_and_measure` returns a flat metrics row used by both runner_offline.py (one pair from
the command line) and run_config.py (a whole config):

  quality     accuracy, macro precision / recall / F1, accuracy before / after the main drift
  detection   per detector (model = detections the model acted on, vocab, adwin):
              detected / missed drifts, false alarms, mean delay
  recovery    instances until accuracy is back to 95% of its pre-drift level, per drift
  system      runtime, throughput, per-batch processing time p50 / p95, RSS growth,
              model size in (word, class) entries
"""
import json
import time
from pathlib import Path

import numpy as np
import psutil

from core.metrics import classification_summary, detection_metrics, prequential, recovery_times

BATCH = 1000  # instances per timed batch (the streaming micro-batch size)


def drift_meta(path: Path) -> dict:
    return json.loads(path.with_name(f"{path.stem}_drift_points.json").read_text())


def default_tolerance(n: int) -> int:
    """Acceptance window after a drift: long enough for a w = 24k (DS) / 4k (scenario)
    vocabulary check to land inside it."""
    return 50_000 if n > 500_000 else 10_000


def default_recovery_window(n: int) -> int:
    return 10_000 if n > 500_000 else 1_000


def model_detections(events: list[dict], detector: str | None) -> list[int]:
    """Change detections of one detector, or (detector=None) every signal the model acted on."""
    if detector is None:
        return [e["idx"] for e in events if e.get("acted")]
    return [e["idx"] for e in events if e["detector"] == detector and e["kind"] == "change"]


def run_and_measure(model, instances: list, meta: dict, tolerance: int | None = None,
                    recovery_window: int | None = None) -> tuple[dict, object]:
    n = len(instances)
    tolerance = tolerance or default_tolerance(n)
    recovery_window = recovery_window or default_recovery_window(n)
    points = meta["drift_points"]
    main_point = meta.get("natural_change_point", points[-1])

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    result = prequential(model, instances, batch_size=BATCH)
    run_s = time.perf_counter() - t0
    rss_after = proc.memory_info().rss

    y, p = result.y_true, result.y_pred
    summary = result.summary()
    pre = classification_summary(y[:main_point], p[:main_point])
    post = classification_summary(y[main_point:], p[main_point:])
    row = {
        "n": n, "n_drifts": len(points), "drift_point": main_point,
        "accuracy": summary["accuracy"], "precision_macro": summary["precision_macro"],
        "recall_macro": summary["recall_macro"], "f1_macro": summary["f1_macro"],
        "accuracy_pre_drift": pre["accuracy"], "accuracy_post_drift": post["accuracy"],
    }

    events = getattr(model, "events", None)
    for name, detector in (("model", None), ("vocab", "vocab"), ("adwin", "adwin")):
        if events is None:
            continue
        det = detection_metrics(model_detections(events, detector), points, tolerance,
                                meta.get("drift_windows"))
        row.update({f"{name}_detections": det["n_detections"], f"{name}_detected": det["detected"],
                    f"{name}_missed": det["missed"], f"{name}_false_alarms": det["false_alarms"],
                    f"{name}_mean_delay": det["mean_delay"]})
    row["rebuilds"] = sum(e["kind"] == "rebuild" for e in events) if events is not None else None

    rec = recovery_times(y == p, points, recovery_window)
    found = [r for r in rec if r is not None]
    row.update({"recovery_mean": float(np.mean(found)) if found else None,
                "recovery_max": max(found) if found else None,
                "not_recovered": sum(r is None for r in rec),
                "recovery_per_drift": json.dumps(rec)})

    batch_ms = np.array(result.batch_seconds) * 1000
    row.update({"runtime_s": run_s, "throughput": n / run_s,
                "batch_ms_p50": float(np.percentile(batch_ms, 50)) if len(batch_ms) else None,
                "batch_ms_p95": float(np.percentile(batch_ms, 95)) if len(batch_ms) else None,
                "rss_growth_mb": (rss_after - rss_before) / 2**20,
                "model_entries": getattr(model, "size", None),
                "tolerance": tolerance, "recovery_window": recovery_window})
    return row, result
