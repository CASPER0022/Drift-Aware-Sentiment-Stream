"""Build DS1: Sentiment140 sorted into true arrival (timestamp) order.

The raw CSV is ordered by label, not by time, so replaying it as-is would be a
single massive artificial drift. This script parses the timestamps, sorts, maps
labels 0/4 -> 0/1 and writes data/processed/ds1.parquet
plus ds1_drift_points.json with the natural change point.

Usage:  python pipeline/build_ds1.py
"""
import zipfile

import pandas as pd

from streams import BASE_COLUMNS, DS1 as OUT, ROOT, save_stream

RAW_DIR = ROOT / "data" / "raw"
RAW_ZIP = RAW_DIR / "trainingandtestdata.zip"
RAW_CSV = RAW_DIR / "training.1600000.processed.noemoticon.csv"

COLUMNS = ["label", "id", "date", "query", "user", "text"]


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        with zipfile.ZipFile(RAW_ZIP) as zf:
            zf.extract(RAW_CSV.name, RAW_DIR)
    return pd.read_csv(RAW_CSV, encoding="latin-1", header=None, names=COLUMNS,
                       dtype={"label": "int8", "id": "int64", "text": "string"})


def build(df: pd.DataFrame) -> pd.DataFrame:
    # Every row is "... PDT 2009"; drop the zone token and treat times as PDT wall clock.
    assert df["date"].str.contains(" PDT ").all(), "unexpected timezone in raw data"
    df["ts"] = pd.to_datetime(df["date"].str.replace(" PDT", "", regex=False),
                              format="%a %b %d %H:%M:%S %Y")
    df["label"] = (df["label"] == 4).astype("int8")  # 0 = negative, 1 = positive

    # Stable sort; ties on the same second are broken by tweet id (monotonic in time).
    df = df.sort_values(["ts", "id"], kind="stable").reset_index(drop=True)
    df.insert(0, "idx", df.index.astype("int64"))  # instance index = time unit t
    return df[BASE_COLUMNS]


def natural_change_point(df: pd.DataFrame) -> int:
    """First instance of the all-negative tail (paper: ~1,326,000)."""
    return int(df.index[df["label"] == 1].max()) + 1


def main() -> None:
    df = build(load_raw())
    cp = natural_change_point(df)
    save_stream(df, OUT, {
        "drift_type": "natural prior shift (stream turns all-negative)",
        "drift_points": [cp],
        "change_ts": str(df["ts"].iat[cp]),
    })

    dup_ids = df["id"].duplicated().sum()
    print(f"  time span : {df['ts'].min()} -> {df['ts'].max()}")
    print(f"  positives : {df['label'].mean():.1%}")
    print(f"  duplicate tweet ids kept: {dup_ids:,} (same text, opposite labels: label noise)")


if __name__ == "__main__":
    main()
