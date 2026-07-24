"""
Day 3 - Language detection and translation for Amharic and Afaan Oromo reviews.

Purpose:
    The original pipeline only understands English. Real customers write
    reviews in Amharic and Afaan Oromo too (we found 68 Amharic reviews
    in the raw data already). This script detects the language of each
    review and translates non-English reviews to English, so they can
    flow through the existing sentiment/theme/risk pipeline.

    Since we know our data realistically contains only three languages
    (English, Amharic, Afaan Oromo), we use a more targeted approach:
        1. Amharic is detected reliably via its unique script (Ge'ez
           alphabet) - no ambiguity possible here.
        2. For remaining Latin-script reviews (English or Afaan Oromo),
           we translate via Google Translate with auto-detection. If the
           translated text comes back nearly identical to the original,
           the review was already English. If it comes back noticeably
           different, it was Afaan Oromo.

Input:
    data/model_features.csv is NOT used here - this script works on the
    original review text, so it reads:
    data/labeled_reviews.csv (has the "review" column)

Output:
    data/labeled_reviews_multilingual.csv
    Same as input, plus:
        - detected_language: "am", "om", or "en"
        - translated_review: English text (identical to original if
          the review was already English)
"""

import difflib
import re
import sys
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator

# Windows terminals default to an older encoding (cp1252) that cannot
# display Amharic script or some special characters. This switches
# console output to UTF-8 so printing (including warning messages that
# quote the original review text) never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path("data/labeled_reviews.csv")
OUTPUT_PATH = Path("data/labeled_reviews_multilingual.csv")

# Ge'ez script Unicode range - covers Amharic and other Ethiopian languages
# that use this alphabet. If any character in the review falls in this
# range, I know for certain the review contains Amharic script.
AMHARIC_SCRIPT_PATTERN = re.compile(r"[\u1200-\u137F]")

# If the translated text is at least this similar to the original text,
# I conclude the review was already in English (translation was a
# no-op). Below this threshold, I conclude it was Afaan Oromo.
ENGLISH_SIMILARITY_THRESHOLD = 0.85


def load_data(path: Path) -> pd.DataFrame:
    """Load the labeled reviews CSV into a DataFrame."""
    return pd.read_csv(path)


def is_amharic_script(text: str) -> bool:
    """Return True if the text contains any Ge'ez (Amharic) script characters."""
    return bool(AMHARIC_SCRIPT_PATTERN.search(str(text)))


def text_similarity(a: str, b: str) -> float:
    """Return a 0-1 similarity score between two strings.

    Used to check whether a "translated" text is essentially unchanged
    from the original (meaning it was already English).
    """
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def safe_translate(text: str, source: str, target: str = "en") -> str:
    """Call GoogleTranslator, but never let a bad review crash the whole run.

    Google Translate occasionally returns None (e.g. for empty text, or
    text it cannot process at all) instead of a translation, and network
    hiccups can raise exceptions. In either case, we fall back to
    returning the original text unchanged, rather than stopping the
    entire batch over one problematic review.
    """
    if not text or not text.strip():
        return text

    try:
        result = GoogleTranslator(source=source, target=target).translate(text)
    except Exception as error:
        print(f"  [warning] translation failed ({type(error).__name__}), keeping original text")
        return text

    if result is None:
        return text

    return result


def detect_and_translate(text: str) -> tuple[str, str]:
    """Detect the language of a single review and translate it to English.

    Returns:
        (detected_language, translated_text)
        detected_language is one of: "am", "om", "en"
    """
    text = str(text)

    if not text.strip():
        return "en", text  # empty/whitespace review - nothing to detect or translate

    if is_amharic_script(text):
        translated = safe_translate(text, source="am")
        return "am", translated

    # Not Amharic script - could be English or Afaan Oromo. Translate
    # with auto-detection and compare to the original to tell them apart.
    translated = safe_translate(text, source="auto")
    similarity = text_similarity(text, translated)

    if similarity >= ENGLISH_SIMILARITY_THRESHOLD:
        return "en", text  # already English, no meaningful translation happened
    else:
        return "om", translated


def process_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Apply language detection and translation to every review in the DataFrame."""
    df = df.copy()

    languages = []
    translations = []

    total = len(df)
    for i, review_text in enumerate(df["review"], start=1):
        language, translated_text = detect_and_translate(review_text)
        languages.append(language)
        translations.append(translated_text)

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} reviews...")

    df["detected_language"] = languages
    df["translated_review"] = translations
    return df


def show_language_summary(df: pd.DataFrame) -> None:
    """Print how many reviews were detected in each language."""
    print()
    print("--- Detected language distribution ---")
    print(df["detected_language"].value_counts())
    print()


def main() -> None:
    df = load_data(DATA_PATH)
    df = process_reviews(df)

    show_language_summary(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved multilingual-processed dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()