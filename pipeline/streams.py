"""Shared helpers for the stream builders: paths, schema and drift-point metadata."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets pipeline scripts import the shared `core` package
    sys.path.insert(0, str(ROOT))
PROCESSED = ROOT / "data" / "processed"
SCENARIOS = ROOT / "data" / "scenarios"
DS1 = PROCESSED / "ds1.parquet"

# Every stream carries at least these columns, in this order; the producer reads them.
BASE_COLUMNS = ["idx", "id", "ts", "label", "user", "text"]


def stream_path(name: str, seed: int = 0) -> Path:
    """ds1 / ds2, or a scenario name such as s_label_flip plus its seed."""
    if name in ("ds1", "ds2"):
        return PROCESSED / f"{name}.parquet"
    return SCENARIOS / f"{name}_seed{seed}.parquet"


def load_ds1(columns=None) -> pd.DataFrame:
    return pd.read_parquet(DS1, columns=columns)


def save_stream(df: pd.DataFrame, path: Path, meta: dict) -> None:
    """Write the stream parquet plus `<stem>_drift_points.json` next to it.

    `meta["drift_points"]` holds instance indices (0-based `idx`) where the true concept
    changes; `meta["drift_windows"]` optionally holds [start, end) spans for gradual drifts.
    """
    assert (df["idx"].to_numpy() == range(len(df))).all(), "idx must be 0..n-1"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {"stream": path.stem, "n": len(df), "positive_share": round(float(df["label"].mean()), 4),
            **meta}
    meta_path = path.with_name(f"{path.stem}_drift_points.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(df):,} tweets, "
          f"{meta['positive_share']:.1%} positive) drift points: {meta['drift_points']}")
