"""ADWIN on the prequential error stream: the drift signal the vocabulary detector lacks.

Each tested instance contributes err = 1 if the prediction was wrong, else 0. ADWIN
(Bifet & Gavalda 2007, river implementation) keeps an adaptive window of this stream and
cuts it when two sub-windows have significantly different means. Only cuts where the error
went *up* count as drift: a falling error means the model is doing better, not worse.
Unlike the vocabulary detector it reacts to changes in P(y|X) even when the words stay
the same (label flips, meaning shifts).
"""
from river import drift


class ErrorADWIN:
    name = "adwin"

    def __init__(self, delta: float = 0.002, clock: int = 32, only_increase: bool = True) -> None:
        self.delta, self.clock, self.only_increase = delta, clock, only_increase
        self.reset()

    def reset(self) -> None:
        """Forget the error history, e.g. after the model was rebuilt."""
        self.adwin = drift.ADWIN(delta=self.delta, clock=self.clock)

    def update(self, err: int) -> bool:
        """Feed one 0/1 error; True if ADWIN detected a rise in the error rate."""
        before = self.adwin.estimation
        self.adwin.update(err)
        if not self.adwin.drift_detected:
            return False
        return not self.only_increase or self.adwin.estimation > before

    @property
    def width(self) -> int:
        """Instances in ADWIN's current window, i.e. those it considers post-drift."""
        return int(self.adwin.width)

    @property
    def error_rate(self) -> float:
        return float(self.adwin.estimation)
