"""
Helper functions for the Streamlit dashboard (app.py).

Purpose:
    Keeps the "business logic" (loading the model, turning a raw review
    into a feature vector) separate from the "display logic" (charts,
    layout) that lives in app.py. This mirrors the same reasoning
    behind config.py and text_utils.py: shared logic belongs in one
    place, not copy-pasted into the UI file.
"""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DATA_PATHS, LANGUAGE_CONFIG

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
SHORT_REVIEW_WORD_THRESHOLD = 3
BANKS = ["Commercial Bank of Ethiopia", "Bank of Abyssinia", "Dashen Bank"]


def load_reviews() -> pd.DataFrame:
    """Load the final, corrected multilingual dataset for the explorer tab."""
    return pd.read_csv(DATA_PATHS.labeled_reviews_multilingual_fixed)


def load_model():
    """Load the trained Random Forest risk model."""
    return joblib.load(DATA_PATHS.risk_model)


def load_feature_columns() -> list[str]:
    """Load the exact feature column names/order the model was trained on."""
    path = DATA_PATHS.model_features.parent / "feature_columns.json"
    with open(path) as f:
        return json.load(f)


def is_amharic_script(text: str) -> bool:
    """Return True if the text contains Amharic (Ge'ez script) characters."""
    pattern = re.compile(LANGUAGE_CONFIG.amharic_script_pattern)
    return bool(pattern.search(str(text)))


def detect_and_translate_live(text: str) -> tuple[str, str]:
    """Detect language and translate a single, freshly typed review.

    Same logic as language_processing.py, reused here so the dashboard
    can handle a brand-new review typed live by a user, not just the
    pre-processed 1,500-review dataset.
    """
    import difflib

    text = str(text)
    if not text.strip():
        return "en", text

    if is_amharic_script(text):
        try:
            translated = GoogleTranslator(source="am", target="en").translate(text)
        except Exception:
            translated = text
        return "am", (translated or text)

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        translated = text
    translated = translated or text

    similarity = difflib.SequenceMatcher(None, text.lower().strip(), translated.lower().strip()).ratio()
    if similarity >= LANGUAGE_CONFIG.english_similarity_threshold:
        return "en", text
    return "om", translated


def classify_sentiment_live(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
    """Classify sentiment of translated text using VADER (same as reprocess_multilingual.py)."""
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return label, compound


def classify_theme_live(text: str) -> str:
    """Assign a theme based on keyword matches (same as reprocess_multilingual.py)."""
    text_lower = str(text).lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return theme
    return DEFAULT_THEME


def build_live_feature_vector(
    translated_text: str,
    detected_language: str,
    theme: str,
    sentiment_score: float,
    bank: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build a single-row feature DataFrame matching the model's training columns exactly.

    Any one-hot column not triggered by this specific review (e.g. a
    different bank or theme) is filled with 0, and the column order is
    forced to match feature_columns exactly - this is what prevents a
    mismatch between how the model was trained and how a live
    prediction is made.
    """
    word_count = len(str(translated_text).split())
    is_short = int(word_count <= SHORT_REVIEW_WORD_THRESHOLD)

    row = {
        "sentiment_score": sentiment_score,
        "is_short_review": is_short,
        "review_word_count": word_count,
    }

    # Initialize every expected one-hot column to 0, then set the ones
    # that actually apply to this review to 1.
    for col in feature_columns:
        if col not in row:
            row[col] = 0

    theme_col = f"identified_theme_{theme}"
    if theme_col in row:
        row[theme_col] = 1

    bank_col = f"bank_{bank}"
    if bank_col in row:
        row[bank_col] = 1

    language_col = f"detected_language_{detected_language}"
    if language_col in row:
        row[language_col] = 1

    # Reindex guarantees column order matches training exactly.
    df = pd.DataFrame([row])
    return df.reindex(columns=feature_columns, fill_value=0)