import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from core.metrics import PrequentialResult, classification_summary


def test_summary_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    y_pred = np.where(rng.random(1000) < 0.8, y_true, 1 - y_true)
    ours = classification_summary(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert np.isclose(ours["accuracy"], accuracy_score(y_true, y_pred))
    assert np.allclose([ours["precision_macro"], ours["recall_macro"], ours["f1_macro"]], [p, r, f])


def test_summary_when_a_class_is_never_predicted():
    ours = classification_summary(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))
    p, r, f, _ = precision_recall_fscore_support([0, 0, 1, 1], [0, 0, 0, 0], average="macro",
                                                 zero_division=0)
    assert np.allclose([ours["precision_macro"], ours["recall_macro"], ours["f1_macro"]], [p, r, f])


def test_windowed_accuracy_keeps_partial_last_window():
    result = PrequentialResult(np.array([1, 1, 1, 1, 1]), np.array([1, 0, 1, 1, 0]))
    ends, acc = result.windowed_accuracy(2)
    assert ends.tolist() == [2, 4, 5]
    assert np.allclose(acc, [0.5, 1.0, 0.0])
