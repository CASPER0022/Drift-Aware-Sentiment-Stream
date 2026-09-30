"""Streaming Multinomial Naive Bayes: accumulative and ageing (Iosifidis et al., Eq. 1-2).

Both models share one interface with every other streaming model in `core/`:
    predict_one(tokens, t) -> int        learn_one(tokens, label, t) -> None
`t` is the model clock (see core.timeunit); the accumulative model ignores it.

AgeingMNB keeps plain accumulated counts plus the last time each class and each
(word, class) pair was observed, and ages them on read (Wagner et al. [21]):
    P(c)   ~ N_c  * exp(-lam * (t - tlo_c))                                   (Eq. 1)
    P(w|c) = N_wc * exp(-lam * (t - tlo_wc)) / sum_j N_jc * exp(-lam * (t - tlo_jc))   (Eq. 2)
with Laplace smoothing alpha added to the numerator and alpha*|V| to the denominator.
Scores are computed in log space. Words never seen in any class are ignored.

Eq. 2's denominator sums over the whole vocabulary. It is kept as a cached per-class
sum S_c = sum_j N_jc * exp(lam * (tlo_jc - t_ref)), so that the denominator at time t is
exp(-lam * (t - t_ref)) * S_c. Learning a word replaces its term in S_c (O(1)); t_ref is
moved forward, rescaling S_c, before exp() can overflow. set_lambda rebuilds S_c exactly,
O(|V|), which is fine because lambda only changes when a drift strategy fires.
"""
import math
from collections import Counter

REBASE_EXPONENT = 30.0  # rebase t_ref once lam * (t - t_ref) exceeds this (e^30 ~ 1e13)


class AccumulativeMNB:
    """accumulativeMNB: grows with the stream and never forgets (lambda = 0)."""

    name = "accumulative"
    lam = 0.0

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_count = [0, 0]
        self.word_count = [{}, {}]  # class -> word -> count
        self.total_words = [0, 0]
        self.vocab: set[str] = set()

    def predict_one(self, tokens: list[str], t: float | None = None) -> int:
        if not self.class_count[0] or not self.class_count[1]:
            return int(self.class_count[1] > self.class_count[0])
        s0, s1 = self.log_scores(tokens)
        return int(s1 > s0)

    def log_scores(self, tokens: list[str]) -> list[float]:
        n = sum(self.class_count)
        v = len(self.vocab)
        counts = Counter(w for w in tokens if w in self.vocab)
        scores = []
        for c in (0, 1):
            wc = self.word_count[c]
            denom = math.log(self.total_words[c] + self.alpha * v)
            s = math.log(self.class_count[c] / n)
            for w, k in counts.items():
                s += k * (math.log(wc.get(w, 0) + self.alpha) - denom)
            scores.append(s)
        return scores

    def learn_one(self, tokens: list[str], label: int, t: float | None = None) -> None:
        self.class_count[label] += 1
        wc = self.word_count[label]
        for w in tokens:
            wc[w] = wc.get(w, 0) + 1
        self.total_words[label] += len(tokens)
        self.vocab.update(tokens)

    @property
    def size(self) -> int:
        """Number of stored (word, class) entries, the memory driver of the model."""
        return len(self.word_count[0]) + len(self.word_count[1])


class AgeingMNB:
    """ageingMNB with a settable ageing factor; lam = 0 reduces to AccumulativeMNB."""

    name = "ageing"

    def __init__(self, lam: float = 0.2, alpha: float = 1.0) -> None:
        self.lam = lam
        self.alpha = alpha
        self.class_count = [0, 0]
        self.class_tlo = [0.0, 0.0]
        self.words = [{}, {}]  # class -> word -> [count, tlo]
        self.vocab: set[str] = set()
        self.t_ref = None  # reference time of the cached sums
        self.t_last = None  # latest time seen in learn_one
        self.norm = [0.0, 0.0]  # S_c, relative to t_ref

    # -- prediction ---------------------------------------------------------------

    def predict_one(self, tokens: list[str], t: float) -> int:
        if not self.class_count[0] or not self.class_count[1]:
            return int(self.class_count[1] > self.class_count[0])
        s0, s1 = self.log_scores(tokens, t)
        return int(s1 > s0)

    def log_scores(self, tokens: list[str], t: float) -> list[float]:
        lam, alpha = self.lam, self.alpha
        av = alpha * len(self.vocab)
        counts = Counter(w for w in tokens if w in self.vocab)
        scores = []
        for c in (0, 1):
            # Eq. 1 (|S| is shared by both classes and drops out of the argmax).
            s = math.log(self.class_count[c]) - lam * (t - self.class_tlo[c])
            if counts:
                denom = math.log(math.exp(-lam * (t - self.t_ref)) * self.norm[c] + av)
                words = self.words[c]
                for w, k in counts.items():
                    entry = words.get(w)
                    num = alpha if entry is None else entry[0] * math.exp(-lam * (t - entry[1])) + alpha
                    s += k * (math.log(num) - denom)
            scores.append(s)
        return scores

    # -- learning -----------------------------------------------------------------

    def learn_one(self, tokens: list[str], label: int, t: float) -> None:
        if self.t_ref is None:
            self.t_ref = t
        elif self.lam * (t - self.t_ref) > REBASE_EXPONENT:
            shrink = math.exp(-self.lam * (t - self.t_ref))
            self.norm = [s * shrink for s in self.norm]
            self.t_ref = t
        self.t_last = t

        self.class_count[label] += 1
        self.class_tlo[label] = t

        lam, t_ref = self.lam, self.t_ref
        words = self.words[label]
        norm = self.norm[label]
        new_w = math.exp(lam * (t - t_ref))
        for w, k in Counter(tokens).items():
            entry = words.get(w)
            if entry is None:
                words[w] = [k, t]
                norm += k * new_w
            else:
                norm += (entry[0] + k) * new_w - entry[0] * math.exp(lam * (entry[1] - t_ref))
                entry[0] += k
                entry[1] = t
        self.norm[label] = max(norm, 0.0)  # guard against rounding just below zero
        self.vocab.update(tokens)

    def set_lambda(self, lam: float) -> None:
        """Change the ageing factor and rebuild the cached normalisers for it."""
        self.lam = lam
        if self.t_last is None:
            return
        self.t_ref = self.t_last
        self.norm = [sum(n * math.exp(lam * (tlo - self.t_ref)) for n, tlo in words.values())
                     for words in self.words]

    @property
    def size(self) -> int:
        return len(self.words[0]) + len(self.words[1])
