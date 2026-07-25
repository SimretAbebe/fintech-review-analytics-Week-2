"""
Tests for text_utils.py

How to run this file:
    pytest tests/test_text_utils.py -v

The "-v" flag means "verbose" - it prints the name of every test and
whether it passed or failed, instead of just a summary count.
"""

import pandas as pd

from text_utils import flag_short_reviews


def test_flag_short_reviews_marks_short_review_as_true():
    """A review with very few words should be flagged as short."""
    df = pd.DataFrame({"review": ["ok"]})

    result = flag_short_reviews(df)

    # "ok" is 1 word, which is <= the default threshold of 3 words,
    # so is_short_review should be True for this row.
    assert result["is_short_review"].iloc[0] == True


def test_flag_short_reviews_marks_long_review_as_false():
    """A review with plenty of words should NOT be flagged as short."""
    df = pd.DataFrame({
        "review": ["This application keeps crashing every time I try to transfer money"]
    })

    result = flag_short_reviews(df)

    # This sentence has more than 3 words, so is_short_review should be False.
    assert result["is_short_review"].iloc[0] == False


def test_flag_short_reviews_counts_words_correctly():
    """review_word_count should match the actual number of words."""
    df = pd.DataFrame({"review": ["good app works well"]})

    result = flag_short_reviews(df)

    # "good app works well" has exactly 4 words.
    assert result["review_word_count"].iloc[0] == 4