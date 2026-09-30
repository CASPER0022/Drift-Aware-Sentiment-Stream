"""EnhancedMNB: the informed ageing MNB plus ADWIN on the prediction error (our contribution).

Two signal channels drive the model:

  vocab channel   every w instances the vocabulary detector checks; its (possibly gated)
                  signal goes to the lambda strategy exactly as in the baseline
  adwin channel   after every instance ADWIN sees err = [prediction != label]; a rise in
                  the error rate triggers the ADWIN reaction

How the channels combine is the fusion mode (the ablation):
  vocab_only   ADWIN is ignored                      (= the baseline InformedAgeingMNB)
  adwin_only   the strategy never sees a change; only ADWIN reacts
  or           both channels act independently
  and          a signal counts only if the other detector fired within `confirm_window`
               instances: a vocab warning/change needs a recent ADWIN rise and vice versa

ADWIN reactions:
  rebuild   replace the model by one trained on the most recent instances: ADWIN's
            post-drift window, at least `min_rebuild`, at most the recent buffer
  fastset   set lambda to lam_max for `fastset_period` instances, then restore it

Each channel has its own cooldown: after a reaction, further triggers *from the same
detector* are ignored for `cooldown` instances, so one drift doesn't trigger twice. The
cooldown is per channel because a shared one lets a false alarm of one detector mask a
real drift seen by the other (S-label-flip: a vocab false alarm at 39,999 swallowed
ADWIN's detection of the flip at 40,063). ADWIN is deliberately *not* reset on a rebuild: a fresh ADWIN has no record of
the model's normal error level, so a drift right after a (false-alarm) rebuild would look
normal to it (seen on S-label-flip: a vocab rebuild at 39,999 blinded ADWIN to the flip at
40,000). A young model's error bump after a rebuild falls inside the cooldown, and ADWIN
drops its old window itself when it fires, so no cascade of rebuilds follows.
"""
from collections import deque

from core.adwin_detector import ErrorADWIN
from core.mnb import AgeingMNB
from core.strategies import Strategy
from core.vocab_detector import CHANGE, NONE, WARNING, VocabularyDetector

FUSION_MODES = ("vocab_only", "adwin_only", "or", "and")
ADWIN_REACTIONS = ("rebuild", "fastset")


class EnhancedMNB:
    name = "enhanced"

    def __init__(self, detector: VocabularyDetector, strategy: Strategy, adwin: ErrorADWIN,
                 fusion: str = "or", adwin_reaction: str = "rebuild",
                 confirm_window: int | None = None, cooldown: int = 2000,
                 min_rebuild: int = 1000, buffer_size: int = 24_000, warmup: int = 1000,
                 fastset_period: int | None = None, lam_max: float = 0.5,
                 alpha: float = 1.0) -> None:
        assert fusion in FUSION_MODES and adwin_reaction in ADWIN_REACTIONS
        self.detector, self.strategy, self.adwin = detector, strategy, adwin
        self.fusion, self.adwin_reaction = fusion, adwin_reaction
        self.confirm_window = confirm_window or detector.w
        self.cooldown, self.min_rebuild, self.warmup = cooldown, min_rebuild, warmup
        self.fastset_period = fastset_period or detector.w
        self.lam_max, self.alpha = lam_max, alpha

        self.model = AgeingMNB(lam=strategy.lam0, alpha=alpha)
        self.recent: deque = deque(maxlen=buffer_size)
        self.n = 0
        self.events: list[dict] = []
        self.lambda_trace: list[tuple[int, float]] = [(0, strategy.lam0)]
        self.checks: list[dict] = []
        self._undrained = 0
        self._last_pred: int | None = None
        self._quiet_until = {"adwin": -1, "vocab": -1}  # cooldown end per channel
        self._last_adwin = -10**12  # index of the latest ADWIN rise
        self._last_vocab = -10**12  # index of the latest vocab warning/change
        self._fastset_until: int | None = None

    @property
    def lam(self) -> float:
        return self.model.lam

    @property
    def size(self) -> int:
        return self.model.size

    def predict_one(self, tokens: list[str], t: float) -> int:
        self._last_pred = self.model.predict_one(tokens, t)
        return self._last_pred

    def learn_one(self, tokens: list[str], label: int, t: float) -> None:
        idx = self.n
        self.n += 1
        err = None if self._last_pred is None else int(self._last_pred != label)
        self._last_pred = None
        self.model.learn_one(tokens, label, t)
        self.recent.append((tokens, label, t))

        if self._fastset_until is not None and idx >= self._fastset_until:
            self._fastset_until = None
            self._set_lambda(idx, self.strategy.lam)

        # ADWIN channel
        # A brand-new model's error swings wildly; ADWIN only starts after `warmup` instances.
        if (self.fusion != "vocab_only" and err is not None and idx >= self.warmup
                and self.adwin.update(err)):
            self._last_adwin = idx
            self.events.append({"idx": idx, "detector": "adwin", "kind": CHANGE,
                                "width": self.adwin.width,
                                "error_rate": round(self.adwin.error_rate, 4)})
            confirmed = self.fusion != "and" or idx - self._last_vocab <= self.confirm_window
            if confirmed and idx >= self._quiet_until["adwin"]:
                self._react_to_adwin(idx)

        # vocab channel
        if self.fusion == "adwin_only":
            return
        result = self.detector.add(tokens, label, t)
        if result is None:
            return
        self.checks.append({"idx": idx, "signal": result.signal,
                            "precision_0": result.precision[0], "precision_1": result.precision[1]})
        signal = result.signal
        if signal != NONE:
            self.events.append({"idx": idx, "detector": "vocab", "kind": signal})
            self._last_vocab = idx
        if self.fusion == "and" and signal in (WARNING, CHANGE):
            signal = CHANGE if idx - self._last_adwin <= self.confirm_window else NONE
        if idx < self._quiet_until["vocab"] and signal == CHANGE:
            signal = NONE  # this detector just reacted; don't stack another reaction on it
        action = self.strategy.on_check(signal)
        if action is None:
            return
        kind, value = action
        if kind == "lambda":
            if self._fastset_until is None:
                self._set_lambda(idx, value)
        elif kind == "rebuild":
            self._rebuild(idx, self.detector.rebuild_instances(), source="vocab")
        if signal == CHANGE:
            self._quiet_until["vocab"] = idx + self.cooldown

    # -- reactions ------------------------------------------------------------------

    def _react_to_adwin(self, idx: int) -> None:
        self._quiet_until["adwin"] = idx + self.cooldown
        if self.adwin_reaction == "rebuild":
            size = max(self.adwin.width, self.min_rebuild)
            self._rebuild(idx, list(self.recent)[-size:], source="adwin")
        else:
            self._fastset_until = idx + self.fastset_period
            self._set_lambda(idx, self.lam_max)

    def _rebuild(self, idx: int, instances: list[tuple], source: str) -> None:
        self.model = AgeingMNB(lam=self.model.lam, alpha=self.alpha)
        for tokens, label, t in instances:
            self.model.learn_one(tokens, label, t)
        self.events.append({"idx": idx, "detector": source, "kind": "rebuild",
                            "rebuild_size": len(instances)})

    def _set_lambda(self, idx: int, lam: float) -> None:
        if lam != self.model.lam:
            self.model.set_lambda(lam)
            self.lambda_trace.append((idx, lam))

    def drain_events(self) -> list[dict]:
        """Events since the previous call, shaped for InfluxSink.write_drift_event."""
        new = self.events[self._undrained:]
        self._undrained = len(self.events)
        return [{"idx": e["idx"], "detector": e["detector"], "kind": e["kind"]} for e in new]
