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


def test_detection_metrics_delay_misses_and_false_alarms():
    from core.metrics import detection_metrics

    m = detection_metrics([500, 10_050, 10_400, 35_000], drift_points=[10_000, 20_000],
                          tolerance=1_000)
    assert m["delays"] == [50, None]  # first hit counts; 20k never detected
    assert (m["detected"], m["missed"]) == (1, 1)
    assert m["false_alarms"] == 2  # 500 and 35,000; the second hit at 10,400 is not false
    assert m["mean_delay"] == 50


def test_detection_metrics_gradual_window_extends_acceptance():
    from core.metrics import detection_metrics

    m = detection_metrics([48_000], drift_points=[30_000], tolerance=1_000,
                          drift_windows=[[30_000, 50_000]])
    assert m["delays"] == [18_000] and m["false_alarms"] == 0


def test_recovery_times():
    from core.metrics import recovery_times

    correct = np.r_[np.ones(5000), np.zeros(1000), np.ones(4000)]  # drop at 5k, back at 6k
    # the window [5975, 6475) holds 25 errors in 500 = 95% accuracy, exactly the target
    assert recovery_times(correct, [5000], window=500) == [975]
    assert recovery_times(np.ones(10_000), [5000], window=500) == [0]  # never dropped
    assert recovery_times(np.r_[np.ones(5000), np.zeros(5000)], [5000], window=500) == [None]
