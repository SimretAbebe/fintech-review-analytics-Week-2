"""
Ethiopian Bank Customer Review Intelligence Platform - Interactive Dashboard.

Run with:
    streamlit run app.py

Two tabs:
    1. Explore Reviews - browse the 1,500 analyzed reviews, filter by
       bank/theme/risk, and see SHAP explanations for real flagged reviews.
    2. Try It Yourself - type any review (English, Amharic, or Afaan
       Oromo) and get a live risk prediction with a plain-language
       explanation of why.
"""

import sys
from pathlib import Path

import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dashboard_utils import (
    BANKS,
    build_live_feature_vector,
    classify_sentiment_live,
    classify_theme_live,
    detect_and_translate_live,
    load_feature_columns,
    load_model,
    load_reviews,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Ethiopian Bank Customer Review System",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_feature_columns():
    return load_feature_columns()


@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()


@st.cache_data
def get_reviews():
    return load_reviews()


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


def explain_prediction_in_words(shap_row: pd.Series, top_n: int = 3) -> list[str]:
    """Turn raw SHAP numbers into plain-English sentences for the UI."""
    sorted_contributions = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
    explanations = []
    for feature, value in sorted_contributions.head(top_n).items():
        direction = "increased" if value > 0 else "decreased"
        clean_name = (
            feature.replace("identified_theme_", "Theme: ")
            .replace("bank_", "Bank: ")
            .replace("detected_language_", "Language: ")
            .replace("_", " ")
        )
        explanations.append(f"**{clean_name}** {direction} the risk score (impact: {value:+.3f})")
    return explanations


st.title(" Ethiopian Bank Customer Review System")
st.caption("CBE, Bank of Abyssinia, Dashen Bank - AI-powered risk detection with multilingual support")

tab_explore, tab_try = st.tabs(["Explore Reviews", "Try It Yourself"])

# ---------------------------------------------------------------------------
# TAB 1: Explore existing reviews
# ---------------------------------------------------------------------------
with tab_explore:
    df = get_reviews()

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_banks = st.multiselect("Bank", options=sorted(df["bank"].unique()), default=list(df["bank"].unique()))
    with col2:
        selected_languages = st.multiselect(
            "Language", options=sorted(df["detected_language"].unique()), default=list(df["detected_language"].unique())
        )
    with col3:
        risk_filter = st.selectbox("Risk status", options=["All", "High-risk only", "Not high-risk only"])

    filtered = df[df["bank"].isin(selected_banks) & df["detected_language"].isin(selected_languages)]
    if risk_filter == "High-risk only":
        filtered = filtered[filtered["is_high_risk"] == 1]
    elif risk_filter == "Not high-risk only":
        filtered = filtered[filtered["is_high_risk"] == 0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviews shown", len(filtered))
    m2.metric("High-risk", int(filtered["is_high_risk"].sum()))
    m3.metric("Avg. rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "\u2013")
    non_english_count = int((filtered["detected_language"] != "en").sum())
    m4.metric("Non-English reviews", non_english_count)

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Theme distribution")
        st.bar_chart(filtered["identified_theme"].value_counts())
    with chart_col2:
        st.subheader("Sentiment distribution")
        st.bar_chart(filtered["sentiment_label"].value_counts())

    st.divider()
    st.subheader("Browse reviews")
    display_cols = ["review", "translated_review", "detected_language", "bank", "rating", "sentiment_label", "identified_theme", "is_high_risk"]
    st.dataframe(filtered[display_cols], use_container_width=True, height=350)

    st.divider()
    st.subheader("Explain a specific flagged review")
    high_risk_reviews = filtered[filtered["is_high_risk"] == 1].reset_index(drop=True)
    if len(high_risk_reviews) == 0:
        st.info("No high-risk reviews match the current filters.")
    else:
        selected_idx = st.selectbox(
            "Pick a high-risk review to explain",
            options=high_risk_reviews.index,
            format_func=lambda i: f"[{high_risk_reviews.loc[i, 'bank']}] {high_risk_reviews.loc[i, 'translated_review'][:80]}...",
        )
        chosen_review = high_risk_reviews.loc[selected_idx]
        st.write(f"**Original text:** {chosen_review['review']}")
        if chosen_review["detected_language"] != "en":
            st.write(f"**Translated:** {chosen_review['translated_review']}")
        st.write(f"**Bank:** {chosen_review['bank']} | **Rating:** {chosen_review['rating']} | **Theme:** {chosen_review['identified_theme']}")

# ---------------------------------------------------------------------------
# TAB 2: Try it yourself - live prediction
# ---------------------------------------------------------------------------
with tab_try:
    st.subheader("Type a review in English, Amharic, or Afaan Oromo")
    st.caption("The system will detect the language, translate if needed, and predict the customer's risk level live.")

    example_col1, example_col2, example_col3 = st.columns(3)
    example_text = None
    if example_col1.button("Try an English example"):
        example_text = "My transfer failed twice this week and customer support never replied"
    if example_col2.button("Try an Amharic example"):
        example_text = "\u12a0\u1275\u122b\u1295\u1235\u134b\u122d \u12a0\u120d\u1273\u12ad\u120d\u121d\u1362 \u1260\u1323\u121d \u12a0\u1235\u127a\u1308\u122d"
    if example_col3.button("Try an Afaan Oromo example"):
        example_text = "Maallaqni koo hin argamne, doggoggora guddaa qaba"

    user_review = st.text_area("Review text", value=example_text or "", height=100)
    selected_bank = st.selectbox("Which bank is this review for?", options=BANKS)

    if st.button("Analyze this review", type="primary"):
        if not user_review.strip():
            st.warning("Please type a review first.")
        else:
            with st.spinner("Detecting language and translating..."):
                detected_language, translated_text = detect_and_translate_live(user_review)

            analyzer = get_analyzer()
            sentiment_label, sentiment_score = classify_sentiment_live(translated_text, analyzer)
            theme = classify_theme_live(translated_text)

            language_names = {"en": "English", "am": "Amharic", "om": "Afaan Oromo"}
            st.success(f"Detected language: **{language_names.get(detected_language, detected_language)}**")

            if detected_language != "en":
                st.write(f"**Translated to English:** {translated_text}")

            info_col1, info_col2 = st.columns(2)
            info_col1.metric("Sentiment", sentiment_label, f"{sentiment_score:+.3f}")
            info_col2.metric("Detected theme", theme)

            model = get_model()
            feature_columns = get_feature_columns()
            feature_vector = build_live_feature_vector(
                translated_text=translated_text,
                detected_language=detected_language,
                theme=theme,
                sentiment_score=sentiment_score,
                bank=selected_bank,
                feature_columns=feature_columns,
            )

            risk_probability = model.predict_proba(feature_vector)[0, 1]
            is_high_risk = risk_probability >= 0.5

            st.divider()
            if is_high_risk:
                st.error(f"**HIGH RISK** predicted risk probability: {risk_probability:.1%}")
            else:
                st.success(f"**NOT HIGH RISK** predicted risk probability: {risk_probability:.1%}")

            explainer = get_shap_explainer(model)
            shap_values = explainer(feature_vector)
            shap_row = pd.Series(shap_values.values[0, :, 1], index=feature_columns)

            st.subheader("Why did the model decide this?")
            for explanation in explain_prediction_in_words(shap_row):
                st.write(f"- {explanation}")