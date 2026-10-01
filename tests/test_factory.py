import pytest

from core.enhanced import EnhancedMNB
from core.factory import MODELS, build_model
from core.informed import InformedAgeingMNB


@pytest.mark.parametrize("name", MODELS)
def test_every_model_builds_and_runs(name):
    model = build_model({"model": name, "lam": 0.1, "w": 10})
    for i in range(30):
        model.predict_one(["good", "day"] if i % 2 else ["bad", "day"], i / 10)
        model.learn_one(["good", "day"] if i % 2 else ["bad", "day"], i % 2, i / 10)
    assert model.predict_one(["good"], 3.0) in (0, 1)


def test_defaults_are_the_frozen_day8_settings():
    enhanced = build_model({"model": "enhanced"})
    assert isinstance(enhanced, EnhancedMNB)
    assert (enhanced.fusion, enhanced.adwin.delta, enhanced.detector.alpha) == ("or", 0.01, 1.8)
    assert enhanced.min_rebuild == 100
    baseline = build_model({"model": "informed", "w": 24_000})
    assert isinstance(baseline, InformedAgeingMNB) and baseline.detector.w == 24_000


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model({"model": "svm"})
