"""Build DS2: a more volatile variant of DS1 (Iosifidis et al. section 5.1).

The paper removes "certain fractions of instances" from DS1 to get 1,073,065 tweets
(378,288 positive, 694,777 negative) but does not say which. We cut the pre-change part
of DS1 into equal segments and keep a different fraction of each class per segment, so
the class prior swings at known segment boundaries. The all-negative tail is kept whole,
which preserves the natural change point. Keep fractions follow the relative patterns
below, scaled so that the class totals match the paper exactly.

Usage:  python pipeline/build_ds2.py [--seed 42]
"""
import argparse

import numpy as np
import pandas as pd

from streams import BASE_COLUMNS, PROCESSED, load_ds1, save_stream

TARGET_POS, TARGET_NEG = 378_288, 694_777
# Relative keep weights per pre-change segment; the swings create the prior shifts.
POS_PATTERN = np.array([1.0, 0.25, 0.8, 0.15, 0.6, 0.3])
NEG_PATTERN = np.array([0.6, 1.0, 0.5, 1.0, 0.7, 1.0])


def keep_counts(sizes: np.ndarray, pattern: np.ndarray, target: int) -> np.ndarray:
    """Integer counts k_i <= sizes_i, proportional to pattern_i * sizes_i, summing to target."""
    lo, hi = 0.0, 1.0 / pattern.min()
    for _ in range(100):  # bisect on the scale factor; min(1, s*w) keeps fractions valid
        s = (lo + hi) / 2
        lo, hi = (s, hi) if (np.minimum(1, s * pattern) * sizes).sum() < target else (lo, s)
    counts = np.floor(np.minimum(1, s * pattern) * sizes).astype(int)
    short = target - counts.sum()
    for i in np.argsort(-(sizes - counts))[:short]:  # hand out the rounding remainder
        counts[i] += 1
    assert counts.sum() == target and (counts <= sizes).all()
    return counts


def build(ds1: pd.DataFrame, change_idx: int, seed: int) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(seed)
    labels = ds1["label"].to_numpy()
    bounds = np.linspace(0, change_idx, len(POS_PATTERN) + 1).astype(int)
    segments = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]

    tail_neg = len(ds1) - change_idx
    keep = [np.arange(change_idx, len(ds1))]
    for cls, pattern, target in ((1, POS_PATTERN, TARGET_POS), (0, NEG_PATTERN, TARGET_NEG - tail_neg)):
        pools = [seg[labels[seg] == cls] for seg in segments]
        counts = keep_counts(np.array([len(p) for p in pools]), pattern, target)
        keep += [rng.choice(p, size=k, replace=False) for p, k in zip(pools, counts)]

    kept = np.sort(np.concatenate(keep))
    ds2 = ds1.iloc[kept].reset_index(drop=True)
    ds2["idx"] = np.arange(len(ds2), dtype="int64")
    # Map each DS1 boundary to its position in DS2 (segment boundaries + natural change).
    drift_points = [int(np.searchsorted(kept, b)) for b in bounds[1:]]
    return ds2[BASE_COLUMNS], drift_points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds1 = load_ds1()
    change_idx = int(ds1.index[ds1["label"] == 1].max()) + 1
    ds2, points = build(ds1, change_idx, args.seed)

    assert len(ds2) == TARGET_POS + TARGET_NEG and ds2["label"].sum() == TARGET_POS
    seg_share = [round(float(s.mean()), 3) for s in np.split(ds2["label"].to_numpy(), points)]
    save_stream(ds2, PROCESSED / "ds2.parquet", {
        "drift_type": "prior shifts at segment boundaries + natural all-negative tail",
        "drift_points": points,
        "natural_change_point": points[-1],
        "segment_positive_share": seg_share,
        "seed": args.seed,
    })
    print(f"  positive share per segment: {seg_share}")


if __name__ == "__main__":
    main()
