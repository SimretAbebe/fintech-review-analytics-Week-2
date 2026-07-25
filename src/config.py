"""
Shared configuration for the risk-labeling and feature pipeline.

Purpose:
    Several constants (the risk-label rule, file paths) were previously
    defined separately in label_risk.py and reprocess_multilingual.py.
    Duplicated constants are a real bug risk: if the risk rule were
    ever changed in one file and not the other, the two scripts would
    silently disagree with each other. This module is the single
    source of truth for both.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RiskLabelConfig:
    """Rule used to build the is_high_risk proxy target label.

    A review is labeled high-risk if its rating is at or below
    max_rating AND its theme is one of risky_themes.
    """

    max_rating: int = 2
    risky_themes: frozenset[str] = field(
        default_factory=lambda: frozenset({"Account Access", "Transaction Performance"})
    )


@dataclass(frozen=True)
class TextConfig:
    """Settings for identifying low-information ("short") reviews."""

    short_review_word_threshold: int = 3


@dataclass(frozen=True)
class LanguageConfig:
    """Settings for language detection and translation."""

    # Ge'ez script Unicode range (covers Amharic).
    amharic_script_pattern: str = r"[\u1200-\u137F]"
    # Translated text at or above this similarity to the original is
    # considered "already English" (no meaningful translation happened).
    english_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class DataPaths:
    """Central file locations used across the pipeline."""

    analyzed_reviews: Path = Path("data/analyzed_reviews.csv")
    labeled_reviews: Path = Path("data/labeled_reviews.csv")
    labeled_reviews_multilingual: Path = Path("data/labeled_reviews_multilingual.csv")
    labeled_reviews_multilingual_fixed: Path = Path("data/labeled_reviews_multilingual_fixed.csv")
    model_features: Path = Path("data/model_features.csv")
    risk_model: Path = Path("models/risk_model.joblib")


# Single shared instances - import these rather than constructing new ones,
# so every script is guaranteed to use the same settings.
RISK_LABEL_CONFIG = RiskLabelConfig()
TEXT_CONFIG = TextConfig()
LANGUAGE_CONFIG = LanguageConfig()
DATA_PATHS = DataPaths()