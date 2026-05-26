"""
Generate a semi-synthetic Amazon Reviews dataset for causal inference.

  X = rating count covariate, binned into num_bins levels
  T = review text
  cond_outcome_xK = LLM-generated P(Y=1 | T=t, X=k)  for k in 0..num_bins-1
  APO = sum_k P(X=k) * cond_outcome_xK  [empirically weighted]
  Y = binary outcome sampled from P(Y=1 | T=t_obs, X=x_obs)

  num_bins=8:
    thresholds = [10, 50, 200, 1000, 5000, 20000, 100000]
    X = 0 (1-10), 1 (11-50), 2 (51-200), 3 (201-1000),
        4 (1001-5000), 5 (5001-20000), 6 (20001-100000), 7 (100000+)

Usage:
  python generate_amazon_reviews.py --output_dir data/amazon --category Electronics --num_bins 8
"""

import argparse
import csv
import gzip
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import wget
from openai import OpenAI


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def get_dataset_files(category: str, data_dir: str) -> tuple[str, str]:
    """Ensure review and metadata files exist; download if missing."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    review_file = path / f"{category}.jsonl.gz"
    meta_file = path / f"meta_{category}.jsonl.gz"

    if not review_file.exists():
        url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz"
        print(f"Downloading reviews: {url}")
        wget.download(url, str(review_file))
        print()

    if not meta_file.exists():
        url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"
        print(f"Downloading metadata: {url}")
        wget.download(url, str(meta_file))
        print()

    return str(review_file), str(meta_file)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

RATING_COUNT_THRESHOLDS = {
    2: [5000],
    4: [50, 1000, 20000],
    8: [10, 50, 200, 1000, 5000, 20000, 100000],
}

META_FIELDS = {
    "rating_number": "x_rating_count",
}

def _is_complete(record: dict) -> bool:
    """Return True if x_rating_count is present and parseable as a positive float."""
    val = record.get("x_rating_count")
    if val is None or val == "" or val == []:
        return False
    try:
        return float(val) > 0
    except (ValueError, TypeError):
        return False


def _extract_meta(meta: dict) -> dict:
    result = {}
    for meta_key, col in META_FIELDS.items():
        val = meta.get(meta_key)
        if val is None or val == [] or val == "" or val == "None":
            result[col] = None
            continue
        if isinstance(val, list):
            val = " | ".join(str(v) for v in val if v)
        elif isinstance(val, dict):
            val = "; ".join(f"{k}: {v}" for k, v in val.items())
        result[col] = val
    return result


def load_reviews(
    category: str,
    data_dir: str,
    max_samples: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """
    Load complete (X, T) pairs from Amazon Reviews 2023.

    Only keeps records where x_rating_count is present and parseable. Stops once
    max_samples complete records have been collected.

    X fields: x_rating_count (used to compute binary covariate x_rating_count_high)
    T field: combined string "Rating: X/5\nReview: ..."
    """
    review_file, meta_file = get_dataset_files(category, data_dir)

    meta_by_asin: dict[str, dict] = {}
    with gzip.open(meta_file, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj.get("parent_asin")
            if asin:
                meta_by_asin[asin] = obj

    records = []
    with gzip.open(review_file, "rt", encoding="utf-8") as f:
        for line in f:
            if len(records) >= max_samples:
                break

            ex = json.loads(line)
            review_text = ex.get("text", "").strip()
            rating = ex.get("rating")
            asin = ex.get("parent_asin")

            if not review_text or len(review_text) < 20 or rating is None or not asin:
                continue

            meta = meta_by_asin.get(asin, {})
            t = f"Rating: {float(rating)}/5\nReview: {review_text}"
            record = {"parent_asin": asin, "t": t}
            record.update(_extract_meta(meta))

            if not _is_complete(record):
                continue

            records.append(record)

    print(f"Loaded {len(records)} complete review records.")
    return records


def assign_bin(rating_count: float, thresholds: list[int]) -> int:
    """Return the bin index for a given rating_count given a list of thresholds."""
    for i, t in enumerate(thresholds):
        if rating_count <= t:
            return i
    return len(thresholds)


def bin_label(bin_idx: int, thresholds: list[int]) -> str:
    """Return a human-readable range string for bin_idx, e.g. '51–200'."""
    if bin_idx == 0:
        return f"1\u2013{thresholds[0]:,}"
    elif bin_idx == len(thresholds):
        return f"more than {thresholds[-1]:,}"
    else:
        return f"{thresholds[bin_idx - 1] + 1:,}\u2013{thresholds[bin_idx]:,}"


def empirical_bin_probs(records: list[dict], thresholds: list[int]) -> list[float]:
    """Compute empirical P(X=k) for each bin k."""
    num_bins = len(thresholds) + 1
    bins = [assign_bin(float(r.get("x_rating_count", 0) or 0), thresholds) for r in records]
    counts = np.bincount(bins, minlength=num_bins)
    probs = counts / len(records)
    print(f"Empirical bin distribution ({num_bins} bins):")
    for k in range(num_bins):
        label = bin_label(k, thresholds)
        print(f"  X={k} ({label}): {counts[k]} records  P={probs[k]:.4f}")
    return probs.tolist()


# ---------------------------------------------------------------------------
# LLM outcome generation
# ---------------------------------------------------------------------------

OUTCOME_SYSTEM_PROMPT = """\
You are a customer shopping for an electronic product online on Amazon. \
You get to see the number of ratings the product has and one complete user review, with their rating. \
Based on both these things, what is the probability (0.0 to 1.0) that you will purchase the product? \
Reason about how the number of ratings and the particular review affect your probability of purchase: \
\n- Larger number of ratings indicate a popular product, making you more likely to purchase it. \
\n- The more positive the review sentiment is, the more likely you are to purchase it. \
\n- A detailed and informative review makes your purchase probability more aligned with the review sentiment. \
\n- A large number of ratings also reinforces the review sentiment and accordingly your purchase probability.
\nRespond with only a single float between 0.0 and 1.0, nothing else."""


def _build_user_prompt(record: dict, label: str) -> str:
    """Build the LLM prompt for a given record and rating count bin label."""
    lines = [
        f"Number of Ratings: {label}",
        f"\n{record['t']}",
    ]
    return "\n".join(lines)


def _query_llm(client: OpenAI, model: str, user_prompt: str, retry_delay: float) -> Optional[float]:
    """Call the LLM and return a float probability, or None on failure."""
    text = ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": OUTCOME_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=20,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            return float(text)
        except (ValueError, AttributeError) as e:
            print(f"  Parse error on attempt {attempt + 1}: {e!r} (got {text!r})")
            return None
        except Exception as e:
            print(f"  API error on attempt {attempt + 1}: {e!r}")
            if attempt < 2:
                time.sleep(retry_delay * (attempt + 1))
    return None


def generate_outcomes(
    records: list[dict],
    writer: csv.DictWriter,
    rng: np.random.Generator,
    output_file,
    bin_probs: list[float],
    thresholds: list[int],
    model: str = "gpt-5.1-2025-11-13",
    batch_size: int = 20,
    retry_delay: float = 2.0,
) -> int:
    """
    For each record, call the LLM once per bin to generate
    cond_outcome_xK = P(Y=1 | T=t, X=k) for k in 0..num_bins-1.
    The true APO is the empirically-weighted average:
      apo = sum_k P(X=k) * cond_outcome_xK
    where P(X=k) = empirical bin probability across the full dataset.
    Y is sampled from the probability matching the observed bin.
    Writes rows to `writer`. Returns number of written rows.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")
    client = OpenAI(api_key=api_key)
    num_bins = len(thresholds) + 1
    kept = 0

    for i in range(0, len(records), batch_size):
        if i == 0:
            print(OUTCOME_SYSTEM_PROMPT)
            for k in range(num_bins):
                print(_build_user_prompt(records[0], bin_label(k, thresholds)))
        batch = records[i : i + batch_size]
        print(f"Generating outcomes for records {i}–{i + len(batch) - 1} / {len(records)}...")

        for record in batch:
            # Determine the observed bin for this record
            try:
                rating_count = float(record.get("x_rating_count", 0) or 0)
            except (ValueError, TypeError):
                rating_count = 0
            actual_bin = assign_bin(rating_count, thresholds)

            # Query LLM for each bin (counterfactual rating count levels)
            probs = []
            for k in range(num_bins):
                prob = _query_llm(client, model, _build_user_prompt(record, bin_label(k, thresholds)), retry_delay)
                if prob is None:
                    raise RuntimeError(f"LLM failed to return a probability for record {record.get('parent_asin')!r}")
                probs.append(prob)

            apo = sum(bin_probs[k] * probs[k] for k in range(num_bins))
            prob_actual = probs[actual_bin]
            y = int(rng.random() < prob_actual)

            write_row(writer, record, probs, apo, y)
            kept += 1

        output_file.flush()

    return kept


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

X_COLS = ["x_rating_count"]
T_COL = "t"
Y_COL = "y_purchase"
APO_COL = "apo"

def get_fieldnames(num_bins: int) -> list[str]:
    cond_cols = [f"cond_outcome_x{k}" for k in range(num_bins)]
    return ["parent_asin"] + X_COLS + [T_COL] + cond_cols + [APO_COL, Y_COL]


def write_row(
    writer: csv.DictWriter,
    record: dict,
    probs: list[float],
    apo: float,
    y: int,
) -> None:
    row = {k: (record[k] if record.get(k) is not None else "") for k in ["parent_asin"] + X_COLS}
    row[T_COL] = record[T_COL]
    for k, prob in enumerate(probs):
        row[f"cond_outcome_x{k}"] = prob
    row[APO_COL] = apo
    row[Y_COL] = y
    writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Amazon Reviews (X, T, Y) dataset.")
    parser.add_argument("--output_dir", type=str, default="data/amazon",
                        help="Directory for the generated CSV output.")
    parser.add_argument("--raw_dir", type=str, default="/mfs1/u/nikita/highdim_apo_data",
                        help="Directory for downloaded raw JSONL files.")
    parser.add_argument("--category", type=str, default="Electronics",
                        help="Amazon product category (e.g. Electronics, Books).")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of reviews to load.")
    parser.add_argument("--model", type=str, default="gpt-5.1-2025-11-13",
                        help="OpenAI model for outcome generation.")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Progress logging batch size.")
    parser.add_argument("--num_bins", type=int, default=2, choices=sorted(RATING_COUNT_THRESHOLDS.keys()),
                        help="Number of X bins.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    thresholds = RATING_COUNT_THRESHOLDS[args.num_bins]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_x{args.num_bins}bin" if args.num_bins != 2 else ""
    output_path = output_dir / f"{args.category}_n{args.max_samples}{suffix}.csv"

    if output_path.exists():
        print(f"Output already exists at {output_path}, skipping generation.")
    else:
        records = load_reviews(
            category=args.category,
            data_dir=args.raw_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        bin_probs = empirical_bin_probs(records, thresholds)
        rng = np.random.default_rng(args.seed)
        fieldnames = get_fieldnames(args.num_bins)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            kept = generate_outcomes(
                records, writer, rng, f,
                bin_probs=bin_probs, thresholds=thresholds,
                model=args.model, batch_size=args.batch_size,
            )
        dropped = len(records) - kept
        print(f"Saved {kept} records to {output_path}" + (f" ({dropped} dropped, missing APO)." if dropped else "."))

    df = pd.read_csv(output_path)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    n = len(num_cols)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    prob_cols = {c for c in num_cols if c.startswith("cond_outcome") or c == APO_COL}
    for ax, col in zip(axes, num_cols):
        data = df[col].dropna()
        if col in prob_cols:
            ax.hist(data, bins=40, range=(0, 1), edgecolor="none")
            ax.set_xlim(0, 1)
        else:
            ax.hist(data, bins=40, edgecolor="none")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    # Hide any unused axes in the last row
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved plots to {plot_path}")


if __name__ == "__main__":
    main()
