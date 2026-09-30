"""Majority-class model: the placeholder that proves the pipeline end to end.

It exposes the interface every streaming model in `core/` follows, so the consumer can
swap it for the MNB models later without changes:
    predict_one(tokens) -> int      learn_one(tokens, label) -> None
"""


class MajorityClass:
    name = "majority"

    def __init__(self) -> None:
        self.counts = [0, 0]

    def predict_one(self, tokens: list[str]) -> int:
        return int(self.counts[1] > self.counts[0])

    def learn_one(self, tokens: list[str], label: int) -> None:
        self.counts[label] += 1
