"""
Add derived treatment columns to an Amazon reviews CSV:
  rating    — string "X/5" extracted from "Rating: X.0/5" at the start of the "t" column
  length    — number of words in the review (split on whitespace)
  sentiment — "very positive" if rating > 4.0, else "negative or neutral"

Usage:
    python -m src.datasets.add_treatment_columns \
        src/datasets/data/amazon/Electronics_n10000_x8bin.csv \
        [--output path/to/output.csv]   # default: overwrites in-place
"""

import argparse
import re

import numpy as np
import pandas as pd


_RATING_RE = re.compile(r"^Rating:\s*([\d.]+)\s*/\s*5", re.IGNORECASE)


def extract_rating(t: str) -> float | None:
    m = _RATING_RE.match(str(t).strip())
    return float(m.group(1)) if m else None


def review_word_count(t: str) -> int:
    return len(str(t).split())


def review_length(word_count: int) -> str:
    return "more than 100 words" if word_count > 100 else "less than 100 words"


def sentiment(rating: float | None) -> str | None:
    if rating is None:
        return None
    return "very positive" if rating > 4.0 else "negative or neutral"


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rating_raw = df["t"].apply(extract_rating)             
    df["rating"] = rating_raw.apply(lambda r: f"{int(r)}/5" if r is not None else None)
    df["number of words"] = df["t"].apply(review_word_count)
    df["length"] = df["number of words"].apply(review_length)
    df["sentiment"] = rating_raw.apply(sentiment)
    return df


def main():
    parser = argparse.ArgumentParser(description="Add rating/length/sentiment columns to an Amazon reviews CSV.")
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = add_columns(df)

    out_path = args.output or args.csv_path
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"  rating: {df['rating'].notna().sum()} non-null, {df['rating'].isna().sum()} missing")
    print(f"  number of words: min={df['number of words'].min()}, max={df['number of words'].max()}, mean={df['number of words'].mean():.1f}")
    print(f"  length: {df['length'].value_counts().to_dict()}")
    print(f"  sentiment: {df['sentiment'].value_counts().to_dict()}")

    if "apo" in df.columns:
        print("\ntrue_apos (mean per-obs apo grouped by treatment value):")
        for col in ["rating", "sentiment", "length"]:
            values = sorted(df[col].dropna().unique())
            apos = np.array([df.loc[df[col] == v, "apo"].mean() for v in values])
            print(f"  {col}:")
            for v, a in zip(values, apos):
                print(f"    {v:>22s}  {a:.4f}")


if __name__ == "__main__":
    main()
