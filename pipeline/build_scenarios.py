"""Build the controlled drift scenarios with known ground-truth drift points.

Real Sentiment140 tweets are sampled into topic pools with keyword filters and then
re-timestamped into a synthetic stream, following Costa et al. Every topic is sampled
50/50 by class unless the scenario changes the prior on purpose, so a topic switch never
also shifts P(y). Tweets are drawn without replacement within one stream.

  s_sudden       topic A -> topic B at K                              (virtual drift, P(X))
  s_gradual      P(topic B) rises linearly 0 -> 1 over [K - W/2, K + W/2)
  s_recurring    A -> B -> A -> B, switching every N/4
  s_label_flip   50% topic A + 50% general; at K the labels of topic A invert. P(X) and
                 P(y) stay the same; only P(y|X) changes (real drift, same vocabulary)
  s_prior_shift  general tweets, 50/50 positive before K and 10/90 after  (P(y))

Topic A = work/school, topic B = music/movies, general = tweets matching neither.
Pool B has only ~21.6k negatives, which caps N at 80k for balanced topics.

Usage:  python pipeline/build_scenarios.py [--seeds 0 1 2 3 4] [--n 80000]
"""
import argparse

import numpy as np
import pandas as pd

from streams import BASE_COLUMNS, SCENARIOS, load_ds1, save_stream
from core.preprocess import tokenize

TOPIC_A = {"work", "working", "job", "office", "boss", "school", "class", "classes", "exam",
           "exams", "homework", "study", "studying", "college", "teacher", "meeting", "finals",
           "lecture", "essay", "revision", "coursework", "shift", "interview"}
TOPIC_B = {"music", "song", "songs", "album", "concert", "band", "movie", "movies", "film",
           "cinema", "listening", "playlist", "radio", "singing", "lyrics", "tickets", "dvd",
           "trailer", "gig"}
TS_START = pd.Timestamp("2009-06-01 00:00:00")  # synthetic clock: one tweet per second
GRADUAL_WIDTH = 0.25  # transition width as a share of N
FLIP_TOPIC_SHARE = 0.5
PRIOR_AFTER = 0.10  # positive share after the prior shift


def topic_pools(ds1: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    """Row positions in DS1 for each topic and class.

    Dropped: tweets with no tokens, and the 1,685 tweet ids that DS1 holds twice with
    opposite labels (same text), which would be pure label noise in a controlled stream.
    """
    toks = ds1["text"].map(lambda t: set(tokenize(t)))
    in_a = toks.map(lambda s: bool(s & TOPIC_A)).to_numpy()
    in_b = toks.map(lambda s: bool(s & TOPIC_B)).to_numpy()
    usable = ((toks.map(len) > 0) & ~ds1["id"].duplicated(keep=False)).to_numpy()
    masks = {"A": in_a & ~in_b & usable, "B": in_b & ~in_a & usable,
             "general": ~in_a & ~in_b & usable}
    labels = ds1["label"].to_numpy()
    return {name: {c: np.flatnonzero(m & (labels == c)) for c in (0, 1)} for name, m in masks.items()}


class Sampler:
    """Draws DS1 rows from the pools without replacement."""

    def __init__(self, pools, rng: np.random.Generator):
        self.rng = rng
        self.queues = {(t, c): rng.permutation(rows) for t, by_cls in pools.items() for c, rows in by_cls.items()}
        self.used = {key: 0 for key in self.queues}

    def take(self, topic: str, cls: int, k: int) -> np.ndarray:
        q, u = self.queues[(topic, cls)], self.used[(topic, cls)]
        if u + k > len(q):
            raise ValueError(f"pool {topic}/{cls} exhausted: need {u + k:,}, have {len(q):,}")
        self.used[(topic, cls)] = u + k
        return q[u:u + k]

    def draw(self, topics: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """One DS1 row per (topic, class) slot, in slot order."""
        rows = np.empty(len(topics), dtype=np.int64)
        for t in np.unique(topics):
            for c in (0, 1):
                slots = np.flatnonzero((topics == t) & (classes == c))
                rows[slots] = self.take(t, c, len(slots))
        return rows


def balanced_classes(n: int, pos_share: float, rng) -> np.ndarray:
    k = round(n * pos_share)
    return rng.permutation(np.r_[np.ones(k, int), np.zeros(n - k, int)])


def classes_per_topic(topics: np.ndarray, rng) -> np.ndarray:
    """50/50 class labels drawn separately within each topic, so P(y | topic) = 0.5."""
    classes = np.empty(len(topics), dtype=int)
    for t in np.unique(topics):
        idx = np.flatnonzero(topics == t)
        classes[idx] = balanced_classes(len(idx), 0.5, rng)
    return classes


def scenario_sudden(n, rng):
    k = n // 2
    topics = np.where(np.arange(n) < k, "A", "B")
    return topics, classes_per_topic(topics, rng), {"drift_type": "sudden", "drift_points": [k]}


def scenario_gradual(n, rng):
    w = int(n * GRADUAL_WIDTH)
    start, end = n // 2 - w // 2, n // 2 + w // 2
    p_b = np.clip((np.arange(n) - start) / (end - start), 0, 1)
    topics = np.where(rng.random(n) < p_b, "B", "A")
    return topics, classes_per_topic(topics, rng), {
        "drift_type": "gradual", "drift_points": [start], "drift_windows": [[start, end]]}


def scenario_recurring(n, rng):
    q = n // 4
    topics = np.array(["A", "B", "A", "B"]).repeat(q)[:n]
    topics = np.r_[topics, np.full(n - len(topics), "B")]
    return topics, classes_per_topic(topics, rng), {
        "drift_type": "recurring", "drift_points": [q, 2 * q, 3 * q]}


def scenario_label_flip(n, rng):
    k = n // 2
    topics = rng.permutation(np.r_[np.full(round(n * FLIP_TOPIC_SHARE), "A"),
                                   np.full(n - round(n * FLIP_TOPIC_SHARE), "general")])
    return topics, classes_per_topic(topics, rng), {
        "drift_type": "real (label flip of topic A, same vocabulary)", "drift_points": [k],
        "flip": {"topic": "A", "from_idx": k}}


def scenario_prior_shift(n, rng):
    k = n // 2
    topics = np.full(n, "general")
    classes = np.r_[balanced_classes(k, 0.5, rng), balanced_classes(n - k, PRIOR_AFTER, rng)]
    return topics, classes, {"drift_type": f"prior shift 50/50 -> {PRIOR_AFTER:.0%} positive",
                             "drift_points": [k]}


SCENARIO_BUILDERS = {
    "s_sudden": scenario_sudden,
    "s_gradual": scenario_gradual,
    "s_recurring": scenario_recurring,
    "s_label_flip": scenario_label_flip,
    "s_prior_shift": scenario_prior_shift,
}


def build(name, ds1, pools, n, seed):
    rng = np.random.default_rng([seed, list(SCENARIO_BUILDERS).index(name)])
    topics, classes, meta = SCENARIO_BUILDERS[name](n, rng)
    rows = Sampler(pools, rng).draw(topics, classes)

    df = ds1.iloc[rows].reset_index(drop=True)
    df = df.rename(columns={"ts": "orig_ts", "label": "orig_label"})
    df["idx"] = np.arange(n, dtype="int64")
    df["ts"] = TS_START + pd.to_timedelta(df["idx"], unit="s")
    df["topic"] = topics
    df["label"] = df["orig_label"]
    if "flip" in meta:
        flipped = (df["topic"] == meta["flip"]["topic"]) & (df["idx"] >= meta["flip"]["from_idx"])
        df.loc[flipped, "label"] = 1 - df.loc[flipped, "label"]
    df["label"] = df["label"].astype("int8")
    return df[BASE_COLUMNS + ["topic", "orig_label", "orig_ts"]], {**meta, "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=80_000)
    ap.add_argument("--only", nargs="+", choices=list(SCENARIO_BUILDERS), default=list(SCENARIO_BUILDERS))
    args = ap.parse_args()

    ds1 = load_ds1(columns=BASE_COLUMNS)
    pools = topic_pools(ds1)
    for t, by_cls in pools.items():
        print(f"pool {t:8s} pos {len(by_cls[1]):>8,}  neg {len(by_cls[0]):>8,}")

    for seed in args.seeds:
        for name in args.only:
            df, meta = build(name, ds1, pools, args.n, seed)
            meta["topics"] = {"A": sorted(TOPIC_A), "B": sorted(TOPIC_B)}
            save_stream(df, SCENARIOS / f"{name}_seed{seed}.parquet", meta)


if __name__ == "__main__":
    main()
