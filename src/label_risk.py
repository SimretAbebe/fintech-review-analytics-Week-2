"""
Day 1 - Build the "is_high_risk" proxy target label.

Purpose:
    The dataset has no ready-made "this customer is at risk of leaving"
    label - it does not exist in the raw data. This script creates one
    using a rule we can defend: a review is labeled high-risk if it has
    a low star rating AND its theme is one associated with serious,
    trust-breaking problems (account access or transaction failures),
    rather than minor annoyances (e.g. UI complaints or feature requests).

    IMPORTANT - avoiding leakage:
    The "rating" column is used ONLY here, to build the label. It must
    NOT be passed into the risk-prediction model later as an input
    feature, since that would let the model "cheat" by reading the
    answer instead of learning from the review's language. See
    README / report for the full explanation.

Input:
    data/processed/analyzed_reviews.csv

Output:
    data/processed/labeled_reviews.csv
    Same columns as the input, plus:
        - review_word_count, is_short_review (from text_utils)
        - is_high_risk (the new target label: 1 = high risk, 0 = not)
"""

from pathlib import Path
import sys

import pandas as pd

from config import DATA_PATHS, RISK_LABEL_CONFIG
from text_utils import flag_short_reviews

# Windows terminals default to an older encoding (cp1252) that cannot
# display emoji or some special characters found in real reviews. This
# switches console output to UTF-8 so printing never crashes on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = DATA_PATHS.analyzed_reviews
OUTPUT_PATH = DATA_PATHS.labeled_reviews


def load_data(path: Path) -> pd.DataFrame:
    """Load the analyzed reviews CSV into a DataFrame."""
    return pd.read_csv(path)


def add_high_risk_label(
    df: pd.DataFrame,
    max_rating: int = RISK_LABEL_CONFIG.max_rating,
    risky_themes: frozenset[str] = RISK_LABEL_CONFIG.risky_themes,
) -> pd.DataFrame:
    """Add the is_high_risk column using the rating + theme rule.

    A review is labeled high-risk (1) if:
        - its rating is <= max_rating, AND
        - its identified_theme is in risky_themes

    All other reviews are labeled 0 (not high-risk).

    Args:
        df: DataFrame with "rating" and "identified_theme" columns.
        max_rating: ratings at or below this count as "low".
        risky_themes: set of theme names considered serious enough to
            count toward the high-risk label.

    Returns:
        A copy of df with a new integer column "is_high_risk" (1 or 0).
    """
    df = df.copy()
    is_low_rating = df["rating"] <= max_rating
    is_risky_theme = df["identified_theme"].isin(risky_themes)
    df["is_high_risk"] = (is_low_rating & is_risky_theme).astype(int)
    return df


def summarize_label_balance(df: pd.DataFrame) -> None:
    """Print how many reviews landed in each class, and the percentage split.

    This check matters: if the label is extremely imbalanced (e.g. 2% or
    98% high-risk), a model could get "high accuracy" just by always
    guessing the majority class, without learning anything useful.
    """
    counts = df["is_high_risk"].value_counts()
    percentages = df["is_high_risk"].value_counts(normalize=True) * 100

    print("--- is_high_risk label balance ---")
    for label in sorted(counts.index):
        name = "High-risk (1)" if label == 1 else "Not high-risk (0)"
        print(f"{name}: {counts[label]} reviews ({percentages[label]:.1f}%)")
    print()


def show_sample_examples(df: pd.DataFrame, n: int = 5) -> None:
    """Print a few example reviews from each class for a manual sanity check.

    This is the step where a human (you) reads the actual reviews and
    confirms the labeling rule "makes sense" before we trust it enough
    to train a model on it.
    """
    columns_to_show = ["review", "rating", "identified_theme", "is_high_risk"]

    print(f"--- {n} example HIGH-RISK reviews ---")
    high_risk_sample = df[df["is_high_risk"] == 1][columns_to_show].head(n)
    print(high_risk_sample.to_string(index=False))
    print()

    print(f"--- {n} example NOT-high-risk reviews ---")
    not_risk_sample = df[df["is_high_risk"] == 0][columns_to_show].head(n)
    print(not_risk_sample.to_string(index=False))
    print()


def main() -> None:
    df = load_data(DATA_PATH)
    df = add_high_risk_label(df)
    df = flag_short_reviews(df)

    summarize_label_balance(df)
    show_sample_examples(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved labeled dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()