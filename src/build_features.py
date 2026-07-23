"""
Day 1 - Build the model-ready feature matrix (X) and target (y).

Purpose:
    Turn labeled_reviews.csv into two things a machine learning model can
    actually train on:
        - X: a table of numeric input features
        - y: the target column, is_high_risk

    IMPORTANT - what is deliberately EXCLUDED from X, and why:
        - "rating": this was used to BUILD the is_high_risk label, so it
          cannot also be an input feature - that would be leakage (the
          model would just learn to read the rating instead of learning
          from the review's language).
        - "review" (raw text): not used directly in this simple version.
          I used derived signals instead (sentiment_score, theme,
          word count), which is a common, defensible modeling choice
          for a first version.
        - "review_id", "date": identifiers/metadata, not predictive
          signals.

    What IS included, and why:
        - sentiment_score: how negative/positive the language is
        - identified_theme (one-hot encoded): what the complaint is about
        - review_word_count: how much the customer wrote
        - is_short_review: flags low-information reviews
        - bank (one-hot encoded): allows the model to learn if risk
          patterns differ by bank

Input:
    data/labeled_reviews.csv

Output:
    data/model_features.csv
    A single table containing X (all feature columns) and y
    (is_high_risk), ready to be split into train/test sets on Day 2 plan.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/labeled_reviews.csv")
OUTPUT_PATH = Path("data/model_features.csv")

# Columns from labeled_reviews.csv that must NOT be used as model inputs.
# "rating" is excluded specifically to avoid leakage (see module docstring).
EXCLUDED_COLUMNS = ["review_id", "review", "date", "rating"]

# The column I am trying to predict.
TARGET_COLUMN = "is_high_risk"

# Columns that need one-hot encoding (turning categories into 0/1 columns).
CATEGORICAL_COLUMNS = ["identified_theme", "bank"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the labeled reviews CSV into a DataFrame."""
    return pd.read_csv(path)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature table from the labeled reviews.

    Steps:
        1. Drop columns that must not be used as inputs (identifiers,
           raw text, and "rating" to avoid leakage).
        2. One-hot encode categorical columns (theme, bank) so the
           model can use them as numeric input.
        3. Convert the is_short_review boolean to 0/1.

    Returns:
        A DataFrame containing only feature columns plus the target
        column (is_high_risk).
    """
    df = df.copy()

    # sentiment_label is dropped too: sentiment_score already captures
    # the same information numerically and more precisely.
    columns_to_drop = EXCLUDED_COLUMNS + ["sentiment_label"]
    df = df.drop(columns=columns_to_drop)

    # Turn "is_short_review" (True/False) into 1/0 so every column is numeric.
    df["is_short_review"] = df["is_short_review"].astype(int)

    # One-hot encode theme and bank: each category becomes its own 0/1 column.
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)

    return df


def show_feature_summary(df: pd.DataFrame) -> None:
    """Print the final list of feature columns and the target column."""
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    print(f"--- Final feature columns ({len(feature_columns)}) ---")
    for col in feature_columns:
        print(f"  - {col}")
    print()
    print(f"--- Target column ---\n  - {TARGET_COLUMN}")
    print()
    print("--- Preview of the feature table ---")
    print(df.head(5).to_string())
    print()


def main() -> None:
    df = load_data(DATA_PATH)
    features_df = build_feature_table(df)

    show_feature_summary(features_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved model-ready feature table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()