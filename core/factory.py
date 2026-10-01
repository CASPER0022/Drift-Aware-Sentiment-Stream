"""Build any streaming model from a flat parameter dict (shared by every runner).

Used by the offline runner, the config / sweep runners and the Spark consumer, so a model
spec means the same thing everywhere. Keys not given fall back to DEFAULTS, which are the
settings frozen on Day 8 (ADWIN delta 0.01, vocabulary alpha 1.8, Rebuild on change).
"""
from core.adwin_detector import ErrorADWIN
from core.enhanced import EnhancedMNB
from core.informed import InformedAgeingMNB
from core.majority import MajorityClass
from core.mnb import AccumulativeMNB, AgeingMNB
from core.strategies import make_strategy
from core.vocab_detector import VocabularyDetector

MODELS = ("majority", "accumulative", "fading", "informed", "enhanced")

DEFAULTS = {
    "alpha": 1.0,  # Laplace smoothing
    "lam": 0.0,  # fading: fixed lambda; informed / enhanced: lambda0
    "strategy": "Rebuild", "lam_max": 0.5, "c": 0.1, "decrease": 0.05,
    "w": 4000, "detect_alpha": 1.8, "detect_beta": 0.334, "detect_history": 20,
    "detect_reference": "previous",
    "fusion": "or", "adwin_delta": 0.01, "adwin_reaction": "rebuild",
    "confirm_window": None, "cooldown": 2000, "min_rebuild": 100,
}


def build_model(spec: dict):
    p = {**DEFAULTS, **{k: v for k, v in spec.items() if v is not None or k == "confirm_window"}}
    model = p["model"]
    if model == "majority":
        return MajorityClass()
    if model == "accumulative":
        return AccumulativeMNB(alpha=p["alpha"])
    if model == "fading":  # fadingMNB: ageing MNB with a fixed lambda (reference model)
        return AgeingMNB(lam=p["lam"], alpha=p["alpha"])
    if model in ("informed", "enhanced"):
        detector = VocabularyDetector(w=p["w"], alpha=p["detect_alpha"], beta=p["detect_beta"],
                                      history=p["detect_history"],
                                      reference=p["detect_reference"])
        strategy = make_strategy(p["strategy"], lam0=p["lam"], lam_max=p["lam_max"], c=p["c"],
                                 decrease=p["decrease"])
        if model == "informed":  # the baseline: vocabulary detector + lambda strategy
            return InformedAgeingMNB(detector, strategy, alpha=p["alpha"])
        return EnhancedMNB(detector, strategy, ErrorADWIN(delta=p["adwin_delta"]),
                           fusion=p["fusion"], adwin_reaction=p["adwin_reaction"],
                           confirm_window=p["confirm_window"], cooldown=p["cooldown"],
                           min_rebuild=p["min_rebuild"], lam_max=p["lam_max"], alpha=p["alpha"])
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
