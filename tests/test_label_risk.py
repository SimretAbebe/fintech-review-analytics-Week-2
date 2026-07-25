"""
Tests for label_risk.py - the is_high_risk proxy label logic.

How to run this file:
    pytest tests/test_label_risk.py -v

Why these tests matter more than the text_utils ones:
    label_risk.py decides which reviews count as "high-risk" - this is
    the single most important rule in the whole project, since every
    later step (features, model training, SHAP) depends on it being
    correct. These tests check the rule directly, using small,
    hand-built examples where we already know the right answer -
    rather than trusting the real 1,500-row dataset by eye every time.
"""

import pandas as pd

from label_risk import add_high_risk_label


def test_low_rating_and_risky_theme_is_high_risk():
    """A 1-star review about Account Access should be labeled high-risk."""
    df = pd.DataFrame({
        "rating": [1],
        "identified_theme": ["Account Access"],
    })

    result = add_high_risk_label(df)

    assert result["is_high_risk"].iloc[0] == 1


def test_low_rating_but_minor_theme_is_not_high_risk():
    """A 1-star review about UI design should NOT be labeled high-risk.

    This is the key design decision from Day 1: a bad rating alone isn't
    enough - the complaint also has to be about something serious
    (account access or transaction failures), not a minor annoyance.
    """
    df = pd.DataFrame({
        "rating": [1],
        "identified_theme": ["UI & Design"],
    })

    result = add_high_risk_label(df)

    assert result["is_high_risk"].iloc[0] == 0


def test_high_rating_and_risky_theme_is_not_high_risk():
    """A 5-star review, even about a serious theme, should NOT be high-risk.

    This checks the other half of the rule: rating matters too, not
    just theme. A happy customer isn't "high-risk" just because their
    review happens to mention a transaction.
    """
    df = pd.DataFrame({
        "rating": [5],
        "identified_theme": ["Transaction Performance"],
    })

    result = add_high_risk_label(df)

    assert result["is_high_risk"].iloc[0] == 0


def test_rating_exactly_at_threshold_counts_as_low():
    """A rating of exactly 2 (the boundary) should still count as low.

    Boundary values are a classic place for off-by-one bugs to hide,
    so this is worth testing explicitly rather than assuming it works.
    """
    df = pd.DataFrame({
        "rating": [2],
        "identified_theme": ["Account Access"],
    })

    result = add_high_risk_label(df)

    assert result["is_high_risk"].iloc[0] == 1


def test_multiple_rows_labeled_independently():
    """Each row should be labeled based on its own rating and theme,
    not affected by other rows in the same DataFrame.
    """
    df = pd.DataFrame({
        "rating": [1, 5, 2, 4],
        "identified_theme": [
            "Account Access",
            "Transaction Performance",
            "UI & Design",
            "Customer Support",
        ],
    })

    result = add_high_risk_label(df)

    # Row 0: low rating + risky theme -> high-risk
    # Row 1: high rating -> not high-risk
    # Row 2: low rating but minor theme -> not high-risk
    # Row 3: high rating -> not high-risk
    assert list(result["is_high_risk"]) == [1, 0, 0, 0]