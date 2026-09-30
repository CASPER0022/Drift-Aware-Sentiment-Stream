"""Reactions to detector signals (Iosifidis et al. section 4.2).

A strategy is called once per detector check (every w instances) with the check's
signal and returns what to do with the model:
    ("lambda", value)   set the ageing factor
    ("rebuild", None)   replace the model by one trained on the detector's buffer
    None                leave the model as it is
Each strategy starts at lam0 ("-Init-" in the paper when lam0 > 0, "-Zero-" when 0).
"""
from core.vocab_detector import CHANGE


class Strategy:
    name = "none"

    def __init__(self, lam0: float) -> None:
        self.lam0 = lam0
        self.lam = lam0

    def on_check(self, signal: str):
        return None

    def _set(self, lam: float):
        if lam == self.lam:
            return None
        self.lam = lam
        return ("lambda", lam)


class SlowIncreaseUpToALimit(Strategy):
    """On change lam += c, capped at lam_max; never decreases."""

    name = "SlowIncreaseUpToALimit"

    def __init__(self, lam0: float, c: float, lam_max: float) -> None:
        super().__init__(lam0)
        self.c, self.lam_max = c, lam_max

    def on_check(self, signal):
        if signal == CHANGE:
            return self._set(min(self.lam + self.c, self.lam_max))
        return None


class SlowIncreaseFastReset(Strategy):
    """On change lam += c up to lam_max; after one window at lam_max, reset to lam0."""

    name = "SlowIncreaseFastReset"

    def __init__(self, lam0: float, c: float, lam_max: float) -> None:
        super().__init__(lam0)
        self.c, self.lam_max = c, lam_max

    def on_check(self, signal):
        if self.lam >= self.lam_max:  # lam_max has now been held for one window of w
            return self._set(self.lam0)
        if signal == CHANGE:
            return self._set(min(self.lam + self.c, self.lam_max))
        return None


class FastSetFastReset(Strategy):
    """On change lam = lam_max; back to lam0 after one quiet window of w instances."""

    name = "FastSetFastReset"

    def __init__(self, lam0: float, lam_max: float) -> None:
        super().__init__(lam0)
        self.lam_max = lam_max

    def on_check(self, signal):
        if signal == CHANGE:
            return self._set(self.lam_max)
        return self._set(self.lam0)


class FastSetSlowDecrease(Strategy):
    """On change lam = lam_max; then lam shrinks by `decrease` (e.g. 5%) per quiet window."""

    name = "FastSetSlowDecrease"

    def __init__(self, lam0: float, lam_max: float, decrease: float) -> None:
        super().__init__(lam0)
        self.lam_max, self.decrease = lam_max, decrease

    def on_check(self, signal):
        if signal == CHANGE:
            return self._set(self.lam_max)
        if self.lam > self.lam0:
            nxt = self.lam * (1 - self.decrease)
            return self._set(nxt if nxt > self.lam0 + 1e-9 else self.lam0)
        return None


class Rebuild(Strategy):
    """Constant lam0; on change rebuild the model from the recent instances."""

    name = "Rebuild"

    def on_check(self, signal):
        return ("rebuild", None) if signal == CHANGE else None


def make_strategy(name: str, lam0: float, lam_max: float = 0.5, c: float = 0.1,
                  decrease: float = 0.05) -> Strategy:
    if name == "SlowIncreaseUpToALimit":
        return SlowIncreaseUpToALimit(lam0, c, lam_max)
    if name == "SlowIncreaseFastReset":
        return SlowIncreaseFastReset(lam0, c, lam_max)
    if name == "FastSetFastReset":
        return FastSetFastReset(lam0, lam_max)
    if name == "FastSetSlowDecrease":
        return FastSetSlowDecrease(lam0, lam_max, decrease)
    if name == "Rebuild":
        return Rebuild(lam0)
    raise ValueError(f"unknown strategy {name!r}")


STRATEGIES = ("SlowIncreaseUpToALimit", "SlowIncreaseFastReset", "FastSetFastReset",
              "FastSetSlowDecrease", "Rebuild")
