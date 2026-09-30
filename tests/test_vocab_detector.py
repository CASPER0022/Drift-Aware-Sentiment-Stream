import numpy as np

from core.informed import InformedAgeingMNB
from core.strategies import make_strategy
from core.vocab_detector import CHANGE, NONE, WARNING, VocabularyDetector


def topic_stream(n: int, switch_at: int | None, seed: int = 0):
    """Tweets drawn from vocabulary A, switching to a disjoint vocabulary B at switch_at."""
    rng = np.random.default_rng(seed)
    word_ids = rng.integers(0, 400, size=(n, 8)).tolist()
    labels = rng.integers(0, 2, size=n).tolist()
    for i in range(n):
        prefix = "b" if switch_at is not None and i >= switch_at else "a"
        yield [f"{prefix}{j}" for j in word_ids[i]], labels[i], i / 1000


def run_detector(stream, **kwargs):
    det = VocabularyDetector(**kwargs)
    return [(i, r.signal) for i, (tok, y, t) in enumerate(stream)
            if (r := det.add(tok, y, t)) is not None]


def test_precision_against_previous_window():
    det = VocabularyDetector(w=2, min_history=99)
    det.add(["x", "y"], 0, 0.0)
    det.add(["z"], 0, 0.0)  # window 1: class 0 vocabulary {x, y, z}
    det.add(["x"], 0, 0.0)
    result = det.add(["q"], 1, 0.0)  # window 2: class 0 {x}
    assert result.precision[0] == 1 / 3  # |{x} ∩ {x, y, z}| / |{x, y, z}|
    assert result.precision[1] is None  # class 1 had no reference yet


def test_stable_stream_raises_no_change():
    signals = run_detector(topic_stream(60_000, None), w=2000)
    assert CHANGE not in [s for _, s in signals]


def test_vocabulary_switch_is_detected_at_the_next_check():
    signals = run_detector(topic_stream(60_000, switch_at=40_000), w=2000)
    changes = [i for i, s in signals if s == CHANGE]
    assert changes and changes[0] == 41_999  # first window that contains the switch


def test_warning_buffers_instances_until_a_quiet_check():
    det = VocabularyDetector(w=1)
    det.history[0].extend([0.5, 0.6, 0.5, 0.6])  # mu = 0.55, sigma = 0.05
    det.reference[0] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
    r = det.add(["a", "b", "c", "d", "e", "f"], 0, 0.0)  # precision 0.6 -> 1 sigma off: warning
    assert r.signal == WARNING and len(det.buffer) == 1
    det.reference[0] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
    det.history[0].clear()
    det.history[0].extend([0.5, 0.6, 0.5, 0.6])
    r = det.add(["a", "b", "c", "d", "e", "f"], 0, 0.0)  # still a warning: buffer grows
    assert r.signal == WARNING and len(det.buffer) == 2
    det.history[0].clear()  # too little history -> quiet check clears the buffer
    r = det.add(["a"], 0, 0.0)
    assert r.signal == NONE and det.buffer == []


def lambdas(strategy, signals):
    out = []
    for s in signals:
        strategy.on_check(s)
        out.append(round(strategy.lam, 4))
    return out


def test_fast_set_fast_reset():
    s = make_strategy("FastSetFastReset", lam0=0.1, lam_max=0.5)
    assert lambdas(s, [NONE, CHANGE, NONE, CHANGE, CHANGE, WARNING]) == [0.1, 0.5, 0.1, 0.5, 0.5, 0.1]


def test_slow_increase_up_to_a_limit():
    s = make_strategy("SlowIncreaseUpToALimit", lam0=0.2, c=0.1, lam_max=0.4)
    assert lambdas(s, [CHANGE, NONE, CHANGE, CHANGE, NONE]) == [0.3, 0.3, 0.4, 0.4, 0.4]


def test_slow_increase_fast_reset():
    s = make_strategy("SlowIncreaseFastReset", lam0=0.2, c=0.2, lam_max=0.5)
    assert lambdas(s, [CHANGE, CHANGE, NONE, CHANGE]) == [0.4, 0.5, 0.2, 0.4]


def test_fast_set_slow_decrease():
    s = make_strategy("FastSetSlowDecrease", lam0=0.1, lam_max=0.5, decrease=0.5)
    assert lambdas(s, [CHANGE, NONE, NONE, NONE, NONE]) == [0.5, 0.25, 0.125, 0.1, 0.1]


def test_rebuild_retrains_on_recent_instances_only():
    model = InformedAgeingMNB(VocabularyDetector(w=2000), make_strategy("Rebuild", lam0=0.2))
    for tokens, label, t in topic_stream(60_000, switch_at=40_000):
        model.learn_one(tokens, label, t)
    rebuilds = [e for e in model.events if e["kind"] == "rebuild"]
    assert rebuilds and rebuilds[0]["idx"] == 41_999
    assert not any(w.startswith("a") for c in (0, 1) for w in model.model.words[c])


def test_drain_events_returns_each_event_once():
    model = InformedAgeingMNB(VocabularyDetector(w=2000),
                              make_strategy("FastSetFastReset", lam0=0.1, lam_max=0.5))
    for tokens, label, t in topic_stream(44_000, switch_at=40_000):
        model.learn_one(tokens, label, t)
    first = model.drain_events()
    assert {"idx": 41_999, "detector": "vocab", "kind": CHANGE} in first
    assert model.drain_events() == []
    assert (41_999, 0.5) in model.lambda_trace
