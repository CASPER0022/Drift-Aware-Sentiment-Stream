"""Prequential (test-then-train) evaluation, classification and drift-detection metrics.

Every instance is first used to test the model and then to train it (Iosifidis et al.
section 5.1). Accuracy over time is reported per tumbling window of `eval_window`
instances, as in the paper's accuracy-over-time plots.

Drift metrics compare detections with the known drift points of a stream. A drift at p
(gradual: spanning [p, e)) counts as detected if a detection falls in its acceptance
interval [p, e + tolerance). The first such detection gives the delay; detections outside
every interval are false alarms. Recovery time is how long after p accuracy needs to get
back to `recovery_ratio` of its pre-drift level.
"""
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class PrequentialResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    batch_seconds: list[float] = field(default_factory=list)  # wall time per batch

    def windowed_accuracy(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        """(end index of each window, accuracy in it); the last partial window is kept."""
        correct = (self.y_true == self.y_pred).astype(float)
        starts = np.arange(0, len(correct), window)
        acc = np.add.reduceat(correct, starts) / np.diff(np.r_[starts, len(correct)])
        return np.minimum(starts + window, len(correct)), acc

    def summary(self) -> dict:
        return classification_summary(self.y_true, self.y_pred)


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy plus per-class and macro precision / recall / F1 for labels {0, 1}.

    A class that is never predicted (or never present) gets precision (recall) 0, which
    is what sklearn reports with zero_division=0.
    """
    out = {"n": int(len(y_true)), "accuracy": float(np.mean(y_true == y_pred))}
    for c in (0, 1):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        pred_c, true_c = int(np.sum(y_pred == c)), int(np.sum(y_true == c))
        p = tp / pred_c if pred_c else 0.0
        r = tp / true_c if true_c else 0.0
        out[f"precision_{c}"], out[f"recall_{c}"] = p, r
        out[f"f1_{c}"] = 2 * p * r / (p + r) if p + r else 0.0
    for m in ("precision", "recall", "f1"):
        out[f"{m}_macro"] = (out[f"{m}_0"] + out[f"{m}_1"]) / 2
    return out


def prequential(model, instances: Iterable[tuple[list[str], int, float]],
                batch_size: int = 1000) -> PrequentialResult:
    """Run test-then-train over (tokens, label, t) triples, timing every `batch_size`."""
    y_true, y_pred, batch_seconds = [], [], []
    t_batch = time.perf_counter()
    for i, (tokens, label, t) in enumerate(instances):
        pred = model.predict_one(tokens, t)
        model.learn_one(tokens, label, t)
        y_true.append(label)
        y_pred.append(pred)
        if (i + 1) % batch_size == 0:
            now = time.perf_counter()
            batch_seconds.append(now - t_batch)
            t_batch = now
    return PrequentialResult(np.array(y_true, dtype=np.int8), np.array(y_pred, dtype=np.int8),
                             batch_seconds)


def detection_metrics(detections: Iterable[int], drift_points: list[int], tolerance: int,
                      drift_windows: list[list[int]] | None = None) -> dict:
    """Delay, misses and false alarms of `detections` against the true drift points."""
    detections = sorted(detections)
    ends = {start: end for start, end in (drift_windows or [])}
    intervals = [(p, ends.get(p, p) + tolerance) for p in drift_points]
    delays = []
    for lo, hi in intervals:
        hit = next((d for d in detections if lo <= d < hi), None)
        delays.append(None if hit is None else hit - lo)
    false_alarms = sum(not any(lo <= d < hi for lo, hi in intervals) for d in detections)
    found = [d for d in delays if d is not None]
    return {"n_detections": len(detections), "detected": len(found),
            "missed": len(delays) - len(found), "false_alarms": false_alarms,
            "mean_delay": float(np.mean(found)) if found else None, "delays": delays}


def recovery_times(correct: np.ndarray, drift_points: list[int], window: int,
                   recovery_ratio: float = 0.95, pre_windows: int = 5) -> list[int | None]:
    """Per drift point: instances until accuracy is back to `recovery_ratio` x pre-drift.

    The pre-drift level is the accuracy over the `pre_windows * window` instances before
    p (not reaching back past the previous drift). Recovery is the offset r >= 0 of the
    first window [p + r, p + r + window) whose accuracy reaches the target, searched up to
    the next drift point; 0 means accuracy never fell below the target. None = never.
    """
    correct = np.asarray(correct, dtype=float)
    csum = np.r_[0.0, np.cumsum(correct)]
    bounds = list(drift_points) + [len(correct)]
    out = []
    for k, p in enumerate(drift_points):
        prev = drift_points[k - 1] if k else 0
        lo = max(prev, p - pre_windows * window)
        if p - lo < window:  # not enough history to define the pre-drift level
            out.append(None)
            continue
        target = recovery_ratio * (csum[p] - csum[lo]) / (p - lo)
        starts = np.arange(p, bounds[k + 1] - window + 1)
        if len(starts) == 0:
            out.append(None)
            continue
        acc = (csum[starts + window] - csum[starts]) / window
        ok = np.flatnonzero(acc >= target)
        out.append(int(starts[ok[0]] - p) if len(ok) else None)
    return out
