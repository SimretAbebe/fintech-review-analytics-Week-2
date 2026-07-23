"""
Day 1 - Exploratory analysis of the analyzed_reviews.csv dataset.

Purpose:
    Before designing the "is_high_risk" proxy label for the risk-prediction
    model, we need to understand the data we already have: how ratings,
    sentiment, and themes are distributed, and whether there are any data
    quality issues (e.g. mislabeled sentiment on very short reviews) that
    would weaken the label if left unaddressed.

Input:
    data/processed/analyzed_reviews.csv
    Expected columns: review_id, review, rating, date, bank,
    sentiment_label, sentiment_score, identified_theme

Output:
    Printed summary statistics to the console. No files are written by
    this script - it is a read-only exploration step.
"""

from pathlib import Path

import pandas as pd

from text_utils import flag_short_reviews

# Path to the analyzed reviews file. Adjust if your data lives elsewhere.
DATA_PATH = Path("data/processed/analyzed_reviews.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load the analyzed reviews CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def show_shape(df: pd.DataFrame) -> None:
    """Print the number of rows and columns in the dataset."""
    print("--- Dataset shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print()


def show_value_counts(df: pd.DataFrame, column: str, title: str) -> None:
    """Print the frequency distribution of a single column."""
    print(f"--- {title} ---")
    print(df[column].value_counts())
    print()


def show_rating_vs_sentiment(df: pd.DataFrame) -> None:
    """Cross-tabulate rating against sentiment label.

    This is the check that reveals whether the sentiment model is
    behaving consistently with star ratings. A 1-star review labeled
    POSITIVE is a red flag worth investigating.
    """
    print("--- Rating vs. sentiment label (cross-tabulation) ---")
    print(pd.crosstab(df["rating"], df["sentiment_label"]))
    print()


def show_missing_values(df: pd.DataFrame) -> None:
    """Print the count of missing values per column."""
    print("--- Missing values per column ---")
    print(df.isnull().sum())
    print()


def show_bank_counts(df: pd.DataFrame) -> None:
    """Print how many reviews belong to each bank."""
    show_value_counts(df, "bank", "Reviews per bank")


def show_short_review_sentiment_check(df: pd.DataFrame) -> None:
    """Show how often short reviews get a POSITIVE label despite a low rating.

    This is the concrete evidence for why short/low-information reviews
    need to be filtered or handled separately before using sentiment as
    a model feature.
    """
    flagged = flag_short_reviews(df)
    suspicious = flagged[
        (flagged["is_short_review"])
        & (flagged["rating"] <= 2)
        & (flagged["sentiment_label"] == "POSITIVE")
    ]
    print("--- Short, low-rated reviews mislabeled as POSITIVE ---")
    print(f"Count: {len(suspicious)}")
    if len(suspicious) > 0:
        print(suspicious[["review", "rating", "sentiment_label", "sentiment_score"]].head(10))
    print()


def main() -> None:
    df = load_data(DATA_PATH)

    show_shape(df)
    show_value_counts(df, "identified_theme", "Theme value counts")
    show_value_counts(df, "rating", "Rating value counts")
    show_value_counts(df, "sentiment_label", "Sentiment label value counts")
    show_rating_vs_sentiment(df)
    show_bank_counts(df)
    show_missing_values(df)
    show_short_review_sentiment_check(df)


if __name__ == "__main__":
    main()