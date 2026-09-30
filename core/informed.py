"""InformedAgeingMNB: ageing MNB whose lambda is tuned by a change detector (the baseline).

After every learned instance the detector is updated; at each of its checks the strategy
decides whether to change lambda or rebuild the model. Every signal and every lambda
value is logged, for the offline results and for Grafana (via drain_events).
"""
from core.mnb import AgeingMNB
from core.strategies import Strategy
from core.vocab_detector import NONE, VocabularyDetector


class InformedAgeingMNB:
    name = "informed"

    def __init__(self, detector: VocabularyDetector, strategy: Strategy, alpha: float = 1.0) -> None:
        self.detector = detector
        self.strategy = strategy
        self.alpha = alpha
        self.model = AgeingMNB(lam=strategy.lam0, alpha=alpha)
        self.n = 0  # instances learned so far
        self.events: list[dict] = []  # every warning / change / rebuild
        self.lambda_trace: list[tuple[int, float]] = [(0, strategy.lam0)]
        self.checks: list[dict] = []  # every detector check with its precisions
        self._undrained = 0

    @property
    def lam(self) -> float:
        return self.model.lam

    @property
    def size(self) -> int:
        return self.model.size

    def predict_one(self, tokens: list[str], t: float) -> int:
        return self.model.predict_one(tokens, t)

    def learn_one(self, tokens: list[str], label: int, t: float) -> None:
        self.model.learn_one(tokens, label, t)
        self.n += 1
        result = self.detector.add(tokens, label, t)
        if result is None:
            return
        idx = self.n - 1  # index of the instance that closed the window
        self.checks.append({"idx": idx, "signal": result.signal,
                            "precision_0": result.precision[0], "precision_1": result.precision[1]})
        if result.signal != NONE:
            self.events.append({"idx": idx, "detector": "vocab", "kind": result.signal})

        action = self.strategy.on_check(result.signal)
        if action is None:
            return
        kind, value = action
        if kind == "lambda":
            self.model.set_lambda(value)
            self.lambda_trace.append((idx, value))
        elif kind == "rebuild":
            instances = self.detector.rebuild_instances()
            self.model = AgeingMNB(lam=self.strategy.lam0, alpha=self.alpha)
            for inst_tokens, inst_label, inst_t in instances:
                self.model.learn_one(inst_tokens, inst_label, inst_t)
            self.events.append({"idx": idx, "detector": "vocab", "kind": "rebuild",
                                "rebuild_size": len(instances)})

    def drain_events(self) -> list[dict]:
        """Events since the previous call, shaped for InfluxSink.write_drift_event."""
        new = self.events[self._undrained:]
        self._undrained = len(self.events)
        return [{"idx": e["idx"], "detector": e["detector"], "kind": e["kind"]} for e in new]
