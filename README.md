![Tests](https://github.com/SimretAbebe/fintech-review-analytics-Week-2/actions/workflows/unittests.yml/badge.svg)

# Ethiopian Bank Customer Review Platform

An explainable AI system that predicts customer risk from Google Play Store reviews of Ethiopian banks (CBE, Bank of Abyssinia, Dashen Bank) — with native support for reviews written in English, Amharic, or Afaan Oromo.

🔗 **[Live Demo](https://ethiopian-bank-customer-review.onrender.com/)**

## Business Problem

Ethiopian banks receive thousands of customer reviews every month on the Google Play Store, but no team has the capacity to read them all manually. As a result, serious problems — failed transfers, login errors — often go unnoticed until many customers are already frustrated or have left. A second, less visible problem compounds this: a meaningful share of these reviews are written in Amharic or Afaan Oromo, and the original English-only analysis pipeline had no way to understand them, silently misreading real customer feedback.

## Solution Overview

This project upgrades an earlier Week 2 review-analytics project from purely descriptive sentiment/theme analysis into a predictive, explainable, multilingual risk-intelligence system:

1. Engineers a proxy target label (`is_high_risk`) from star rating and complaint theme, since no direct churn label exists in the raw data — with rating deliberately excluded from the model's inputs to avoid data leakage
2. Trains and compares classification models, selecting the one that catches every actual high-risk review in testing
3. Adds SHAP explainability so every prediction can be traced back to specific contributing factors
4. Detects and translates Amharic and Afaan Oromo reviews, correcting a measured bias in the original pipeline
5. Serves everything through a live, interactive dashboard, containerized with Docker and deployed publicly

## Key Results

- **Model recall: 100%** — the final model (Random Forest) caught every genuinely high-risk review in the held-out test set
- **ROC-AUC: 0.982**, precision: 76.2%
- **98 non-English reviews found** (68 Amharic, 30 Afaan Oromo) that the original pipeline had no way to understand
- **86% of those non-English reviews had been mislabeled NEGATIVE** by the original English-only sentiment model, some with confidence as low as 9.5% — corrected by re-analyzing them via translation
- **+5 additional at-risk customers** identified after the multilingual correction, previously invisible to the original system

## Quick Start

```bash
git clone https://github.com/SimretAbebe/Fintech-Review-Analytics
cd Fintech-Review-Analytics
pip install -r requirements.txt
streamlit run app.py
```

**Or with Docker:**

```bash
docker compose up --build
```

Then open `http://localhost:7860`.

## Project Structure

```
Fintech-Review-Analytics/
├── .github/                  # CI workflow (lint + test on every push)
├── .streamlit/                # Dashboard theme config
├── data/                      # Processed datasets (raw data git-ignored)
├── models/                    # Trained model artifact
├── notebooks/                 # Exploratory analysis
├── scripts/                   # Standalone utility scripts
├── src/                       # Pipeline: labeling, features, model, SHAP,
│                               #   multilingual processing, dashboard logic
├── tests/                     # pytest unit tests
├── app.py                     # Streamlit dashboard entry point
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example                # Template for any local environment variables
├── .gitignore
├── requirements.txt            # Full pipeline dependencies
└── requirements-docker.txt     # Minimal runtime dependencies for deployment
```

## Demo

🔗 **[Live Dashboard](https://ethiopian-bank-customer-review.onrender.com/)**

Two tabs:

- **Explore Reviews** — filter by bank, language, and risk status; view theme/sentiment charts
- **Try It Yourself** — type a review in English, Amharic, or Afaan Oromo and get a live, explained risk prediction

## Technical Details

- **Data**: 1,500 Google Play Store reviews (500 each for CBE, BOA, Dashen Bank), scraped and preprocessed in the original Week 2 project
- **Model**: Random Forest classifier, 14 engineered features (sentiment score, theme, bank, detected language, review length), star rating deliberately excluded to prevent leakage
- **Multilingual handling**: Amharic detected via Ge'ez Unicode script (fully reliable); Afaan Oromo distinguished from English via a translation-difference heuristic, after three general-purpose language libraries (langdetect, langid, lingua) were tested and found unable to support it
- **Explainability**: SHAP (TreeExplainer), both global feature importance and per-prediction local explanations
- **Evaluation**: Stratified 80/20 train/test split, fixed random seed; accuracy, precision, recall, F1, ROC-AUC on held-out data

## Future Improvements

- Replace the rule-based proxy label with real churn/complaint-escalation outcomes if such data becomes available
- Swap the translation-difference Afaan Oromo heuristic for a dedicated low-resource language model, if one becomes practically available
- Add persistent storage so the dashboard doesn't need to reload the dataset on every cold start
- Upgrade to a paid hosting tier to remove the free-tier sleep/cold-start delay

## Author

Simret Abebe
