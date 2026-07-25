"""
Day 3 - Fix sentiment and theme labels for non-English reviews.

Purpose:
    I proved that the original English-only sentiment model mislabeled
    Amharic reviews - e.g. a clearly positive Amharic review ("It's a
    great app, I find it impressive") was labeled NEGATIVE with only
    9.5% confidence, because the model could not read the words at all.
    97% (66/68) of Amharic reviews were labeled NEGATIVE - a strong sign
    of blind guessing, not real understanding. Theme extraction had the
    same problem: 64/68 Amharic reviews were dumped into the generic
    "Feature Request / General" bucket, since keyword matching only
    recognizes English words.

    This script re-analyzes ONLY the non-English reviews (98 of them:
    68 Amharic + 30 Afaan Oromo), using their English translations,
    so their sentiment and theme labels are actually meaningful.

    English reviews (1,402 of them) are left untouched, since they were
    already analyzed correctly the first time - no need to redo that
    work.

Tool choice - VADER instead of DistilBERT:
    The original project's brief allowed VADER as a lighter alternative
    to the DistilBERT transformer model. I used VADER here specifically
    because it has no heavy dependencies (DistilBERT requires the torch 
    library, a multi-gigabyte install), while still being a legitimate,
    commonly used sentiment tool. This is documented as a deliberate
    trade-off, not an oversight.

Theme classification:
    A simple keyword-based classifier, matching the same five themes
    used in the original project (Account Access, Transaction
    Performance, UI & Design, Customer Support, Feature Request /
    General), applied to the translated English text.

Input:
    data/labeled_reviews_multilingual.csv

Output:
    data/labeled_reviews_multilingual_fixed.csv
    Same as input, but sentiment_label, sentiment_score, and
    identified_theme are corrected for the 98 non-English reviews.
    is_high_risk is recalculated using the corrected theme + rating.
"""

from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DATA_PATHS, RISK_LABEL_CONFIG

DATA_PATH = DATA_PATHS.labeled_reviews_multilingual
OUTPUT_PATH = DATA_PATHS.labeled_reviews_multilingual_fixed

# Keyword groups for the simple theme classifier. A review is assigned
# to the first theme whose keywords appear in its (translated) text;
# if none match, it falls back to "Feature Request / General".
THEME_KEYWORDS = {
    "Account Access": [
        "login", "log in", "password", "sign in", "account access",
        "authentication", "otp", "verify", "verification", "locked out",
    ],
    "Transaction Performance": [
        "transfer", "transaction", "payment", "balance", "send money",
        "deposit", "withdraw", "slow", "failed", "loading",
    ],
    "Customer Support": [
        "support", "customer service", "response", "complaint", "contact",
        "help desk", "no reply",
    ],
    "UI & Design": [
        "design", "interface", "layout", "color", "dark mode", "ui",
        "look", "appearance",
    ],
}
DEFAULT_THEME = "Feature Request / General"


def load_data(path: Path) -> pd.DataFrame:
    """Load the multilingual-processed reviews CSV."""
    return pd.read_csv(path)


def classify_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
    """Classify sentiment using VADER.

    Returns:
        (sentiment_label, sentiment_score)
        sentiment_label is one of: POSITIVE, NEGATIVE, NEUTRAL
        sentiment_score is VADER's "compound" score (-1 to 1), NOT the
        same 0-1 confidence scale the original DistilBERT scores used -
        this difference is intentional and documented above.
    """
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return label, compound


def classify_theme(text: str) -> str:
    """Assign a theme based on keyword matches in the (translated) text."""
    text_lower = str(text).lower()

    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return theme

    return DEFAULT_THEME


def reprocess_non_english_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run sentiment and theme analysis for non-English reviews only.

    English reviews are left exactly as they were - they don't need
    reprocessing since the original pipeline handled them correctly.
    """
    df = df.copy()
    analyzer = SentimentIntensityAnalyzer()

    non_english_mask = df["detected_language"] != "en"
    non_english_count = non_english_mask.sum()
    print(f"Reprocessing {non_english_count} non-English reviews...")

    for idx in df[non_english_mask].index:
        translated_text = df.loc[idx, "translated_review"]

        sentiment_label, sentiment_score = classify_sentiment(translated_text, analyzer)
        theme = classify_theme(translated_text)

        df.loc[idx, "sentiment_label"] = sentiment_label
        df.loc[idx, "sentiment_score"] = sentiment_score
        df.loc[idx, "identified_theme"] = theme

    return df


def recalculate_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute is_high_risk using the corrected theme + original rating.

    Same rule as Day 1's label_risk.py, reapplied now that theme labels
    for non-English reviews have been corrected.
    """
    df = df.copy()
    is_low_rating = df["rating"] <= RISK_LABEL_CONFIG.max_rating
    is_risky_theme = df["identified_theme"].isin(RISK_LABEL_CONFIG.risky_themes)
    df["is_high_risk"] = (is_low_rating & is_risky_theme).astype(int)
    return df


def show_before_after_comparison(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Show how sentiment and theme labels changed for non-English reviews."""
    non_english_mask = before["detected_language"] != "en"

    print()
    print("--- Sentiment label BEFORE fix (non-English reviews only) ---")
    print(before[non_english_mask]["sentiment_label"].value_counts())
    print()
    print("--- Sentiment label AFTER fix (non-English reviews only) ---")
    print(after[non_english_mask]["sentiment_label"].value_counts())
    print()
    print("--- Theme BEFORE fix (non-English reviews only) ---")
    print(before[non_english_mask]["identified_theme"].value_counts())
    print()
    print("--- Theme AFTER fix (non-English reviews only) ---")
    print(after[non_english_mask]["identified_theme"].value_counts())
    print()
    print("--- is_high_risk count BEFORE fix (whole dataset) ---")
    print(before["is_high_risk"].value_counts())
    print()
    print("--- is_high_risk count AFTER fix (whole dataset) ---")
    print(after["is_high_risk"].value_counts())
    print()


def main() -> None:
    df_before = load_data(DATA_PATH)

    df_after = reprocess_non_english_reviews(df_before)
    df_after = recalculate_risk_label(df_after)

    show_before_after_comparison(df_before, df_after)

    df_after.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved corrected dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()