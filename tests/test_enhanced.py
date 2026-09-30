import numpy as np

from core.adwin_detector import ErrorADWIN
from core.enhanced import EnhancedMNB
from core.informed import InformedAgeingMNB
from core.metrics import prequential
from core.strategies import make_strategy
from core.vocab_detector import VocabularyDetector


def label_flip_stream(n: int, flip_at: int | None, seed: int = 0):
    """Class-dependent words (learnable), every label inverted from flip_at on."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).tolist()
    own = rng.integers(0, 50, size=(n, 4)).tolist()
    shared = rng.integers(0, 200, size=(n, 4)).tolist()
    for i in range(n):
        y = labels[i]
        tokens = [f"{'p' if y else 'n'}{j}" for j in own[i]] + [f"s{j}" for j in shared[i]]
        label = 1 - y if flip_at is not None and i >= flip_at else y
        yield tokens, label, i / 3600  # one tweet per second, t in hours


def error_stream(n: int, p_before: float, p_after: float, change_at: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [int(rng.random() < (p_before if i < change_at else p_after)) for i in range(n)]


def make_enhanced(fusion: str, w: int = 2000, **kwargs) -> EnhancedMNB:
    return EnhancedMNB(VocabularyDetector(w=w), make_strategy("Rebuild", lam0=0.0), ErrorADWIN(),
                       fusion=fusion, **kwargs)


def test_adwin_flags_a_rise_in_error():
    adwin = ErrorADWIN()
    hits = [i for i, e in enumerate(error_stream(20_000, 0.2, 0.5, 10_000)) if adwin.update(e)]
    assert hits and 10_000 <= hits[0] < 10_500


def test_adwin_ignores_a_fall_in_error():
    adwin = ErrorADWIN()
    assert not any(adwin.update(e) for e in error_stream(20_000, 0.5, 0.2, 10_000))


def test_vocab_only_equals_the_baseline():
    stream = list(label_flip_stream(12_000, flip_at=6_000))
    baseline = prequential(InformedAgeingMNB(VocabularyDetector(w=2000),
                                             make_strategy("Rebuild", lam0=0.0)), stream)
    enhanced = prequential(make_enhanced("vocab_only"), stream)
    assert np.array_equal(baseline.y_pred, enhanced.y_pred)


def test_adwin_only_catches_label_flip_and_recovers():
    stream = list(label_flip_stream(12_000, flip_at=6_000))
    model = make_enhanced("adwin_only")
    result = prequential(model, stream)
    adwin_changes = [e["idx"] for e in model.events if e["detector"] == "adwin" and e["kind"] == "change"]
    assert adwin_changes and 6_000 <= adwin_changes[0] < 6_100
    assert any(e["kind"] == "rebuild" and e["detector"] == "adwin" for e in model.events)
    assert not any(e["detector"] == "vocab" for e in model.events)  # vocab channel unused
    _, acc = result.windowed_accuracy(1000)
    assert acc[-1] > 0.9  # recovered on the flipped concept

    accumulative = prequential(make_enhanced("vocab_only", w=10**9), stream)
    assert accumulative.windowed_accuracy(1000)[1][-1] < 0.5  # never detects, stays wrong


def test_and_fusion_needs_both_detectors():
    # With w huge the vocabulary detector never checks, so ADWIN alone must not act.
    model = make_enhanced("and", w=10**9)
    prequential(model, label_flip_stream(12_000, flip_at=6_000))
    assert any(e["detector"] == "adwin" for e in model.events)
    assert not any(e["kind"] == "rebuild" for e in model.events)


def test_no_adwin_signal_during_warmup():
    model = make_enhanced("adwin_only", warmup=5_000)
    prequential(model, label_flip_stream(4_000, flip_at=1_500))
    assert model.events == []


def test_drain_events_returns_each_event_once():
    model = make_enhanced("or")
    prequential(model, label_flip_stream(12_000, flip_at=6_000))
    first = model.drain_events()
    assert any(e["detector"] == "adwin" for e in first)
    assert set(first[0]) == {"idx", "detector", "kind"}
    assert model.drain_events() == []
