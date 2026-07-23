"""
Shared text utility functions used across the review-analysis pipeline.

Keeping small, reusable pieces of logic here (instead of copy-pasting them
into every script) means I only have to fix a bug or change a rule in
one place.
"""

import pandas as pd

# A review shorter than this many words is considered "low-information"
# (e.g. a single word, a username, a broken link) and is likely to confuse
# the sentiment model.
SHORT_REVIEW_WORD_THRESHOLD = 3


def flag_short_reviews(
    df: pd.DataFrame, word_threshold: int = SHORT_REVIEW_WORD_THRESHOLD
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