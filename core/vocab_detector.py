"""Vocabulary-based change detector (Iosifidis et al. section 4.1, Eq. 3-4).

Per class c, every `w` instances the detector compares the vocabulary of the class in
the latest window, V_sl, with the vocabulary V of a reference window of past data:

    precision_c = |V_sl ∩ V| / |V|                                            (Eq. 3)

The value is compared with the moving average mu and standard deviation sigma of the
last `history` checks of that class:

    CHANGE   if |precision - mu| > alpha * sigma                               (Eq. 4)
    WARNING  if |precision - mu| > beta * sigma   (beta < alpha)

A check reports the stronger signal over both classes. From a warning on, instances are
buffered for a possible rebuild; the buffer is dropped when the next check is quiet.
Choices the paper leaves open (document in the report): mu/sigma are over the last
`history` checks, at least `min_history` checks are needed before signalling, and a
class's history restarts after a change so the new regime builds its own baseline.

Reference window: by default the previous window of the same size ("previous"). An
accumulated reference of the whole stream so far ("accumulated") keeps growing while
V_sl does not, so precision falls steadily even on a stable stream (S-sudden: 0.45 ->
0.10 over 80k tweets) and the moving-average test fires constantly.
"""
from collections import deque
from dataclasses import dataclass
from statistics import fmean, pstdev

NONE, WARNING, CHANGE = "none", "warning", "change"


@dataclass
class CheckResult:
    signal: str
    precision: tuple[float | None, float | None]  # per class; None if not yet defined


class VocabularyDetector:
    def __init__(self, w: int = 24_000, alpha: float = 1.8, beta: float = 0.334,
                 history: int = 20, min_history: int = 3, reference: str = "previous") -> None:
        assert beta < alpha and reference in ("previous", "accumulated")
        self.w, self.alpha, self.beta = w, alpha, beta
        self.reference_mode = reference
        self.min_history = min_history
        self.reference: list[set[str]] = [set(), set()]
        self.window: list[set[str]] = [set(), set()]
        self.history = [deque(maxlen=history), deque(maxlen=history)]
        self.in_window = 0
        self.window_instances: list[tuple] = []  # the current window, for rebuilds
        self.buffer: list[tuple] = []  # instances since the last warning started
        self.last_window: list[tuple] = []  # the window evaluated by the latest check
        self.warning_active = False

    def add(self, tokens: list[str], label: int, t: float) -> CheckResult | None:
        """Record one labelled instance; returns a CheckResult every `w` instances."""
        self.window[label].update(tokens)
        self.window_instances.append((tokens, label, t))
        self.in_window += 1
        if self.in_window < self.w:
            return None
        return self._check()

    def _check(self) -> CheckResult:
        signals, precisions = [], []
        for c in (0, 1):
            ref, win, hist = self.reference[c], self.window[c], self.history[c]
            if not ref:  # nothing to compare against yet
                precisions.append(None)
                signals.append(NONE)
            else:
                p = len(win & ref) / len(ref)
                precisions.append(p)
                signals.append(self._classify(p, hist))
                if signals[-1] == CHANGE:
                    hist.clear()
                hist.append(p)
            if self.reference_mode == "previous":
                self.reference[c] = win
            else:
                ref |= win

        signal = CHANGE if CHANGE in signals else WARNING if WARNING in signals else NONE
        if signal == NONE:
            self.buffer, self.warning_active = [], False
        else:
            if not self.warning_active:
                self.buffer = []
            self.buffer.extend(self.window_instances)
            self.warning_active = signal == WARNING

        self.last_window = self.window_instances
        self.window = [set(), set()]
        self.window_instances = []
        self.in_window = 0
        return CheckResult(signal, (precisions[0], precisions[1]))

    def _classify(self, p: float, hist: deque) -> str:
        if len(hist) < self.min_history:
            return NONE
        mu, sigma = fmean(hist), pstdev(hist)
        dev = abs(p - mu)
        if dev <= 1e-12:
            return NONE
        if dev > self.alpha * sigma:
            return CHANGE
        if dev > self.beta * sigma:
            return WARNING
        return NONE

    def rebuild_instances(self) -> list[tuple]:
        """Instances to rebuild from after a change: the warning buffer, which always
        ends with the window that triggered the change."""
        return self.buffer or self.last_window
