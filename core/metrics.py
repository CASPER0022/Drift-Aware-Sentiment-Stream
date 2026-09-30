"""Prequential (test-then-train) evaluation and classification metrics.

Every instance is first used to test the model and then to train it (Iosifidis et al.
section 5.1). Accuracy over time is reported per tumbling window of `eval_window`
instances, as in the paper's accuracy-over-time plots.
"""
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass
class PrequentialResult:
    y_true: np.ndarray
    y_pred: np.ndarray

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
                on_instance: Callable[[int, int, int], None] | None = None) -> PrequentialResult:
    """Run test-then-train over (tokens, label, t) triples.

    `on_instance(i, y_true, y_pred)` is called after each instance, e.g. to feed a drift
    detector with the error signal.
    """
    y_true, y_pred = [], []
    for i, (tokens, label, t) in enumerate(instances):
        pred = model.predict_one(tokens, t)
        model.learn_one(tokens, label, t)
        y_true.append(label)
        y_pred.append(pred)
        if on_instance is not None:
            on_instance(i, label, pred)
    return PrequentialResult(np.array(y_true, dtype=np.int8), np.array(y_pred, dtype=np.int8))
