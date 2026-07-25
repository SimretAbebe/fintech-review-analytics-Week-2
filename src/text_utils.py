"""
Shared text utility functions used across the review-analysis pipeline.

Keeping small, reusable pieces of logic here (instead of copy-pasting them
into every script) means I only have to fix a bug or change a rule in
one place.
"""

import pandas as pd

from config import TEXT_CONFIG


def flag_short_reviews(
    df: pd.DataFrame, word_threshold: int = TEXT_CONFIG.short_review_word_threshold
) -> pd.DataFrame:
    """Add word-count and short-review flag columns to a reviews DataFrame.

    Args:
        df: DataFrame containing a "review" text column.
        word_threshold: reviews with this many words or fewer are flagged
            as short/low-information.

    Returns:
        A copy of df with two new columns added:
            - review_word_count: number of words in the review
            - is_short_review: True if the review is at or below the
              word threshold
    """
    df = df.copy()
    word_counts = df["review"].astype(str).str.split().str.len()
    df["review_word_count"] = word_counts
    df["is_short_review"] = word_counts <= word_threshold
    return df