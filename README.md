![Tests](https://github.com/SimretAbebe/fintech-review-analytics-Week-2/actions/workflows/unittests.yml/badge.svg)
# Fintech Review Analytics

## Project Overview
This project is an end-to-end data analytics pipeline designed for **Omega Consultancy**. It analyzes user reviews from the Google Play Store for three major Ethiopian banks: **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank**.

### Business Need
Mobile banking adoption in Ethiopia is accelerating, and customer reviews have become one of the richest signals of product quality. Left unanalyzed, this feedback is noise. Systematically processed, it becomes a competitive intelligence asset for bank product teams to improve retention and performance.

### Business Scenarios
1.  **Retaining Users:** Analyzing systemic issues like slow loading or transfer failures across all three apps.
2.  **Enhancing Features:** Extracting desired features (e.g., fingerprint login, budgeting tools) to guide development.
3.  **Managing Complaints:** Clustering recurring complaints (e.g., "login error", "OTP not received") to guide customer support prioritization.

---

## Project Structure
```text
fintech-review-analytics/
├── .github/workflows/    # CI/CD (GitHub Actions)
├── data/
│   ├── raw/             # Raw scraped CSV data
│   ├── plots/           # Generated visualizations
│   └── analyzed_reviews.csv # Final processed data
├── notebooks/           # Research and EDA notebooks
│   └── task4_insights.ipynb  # Task 4: Visualizations
├── scripts/             # Utility scripts
├── src/                 # Core source code
│   ├── scraper.py           # Task 1: Data collection
│   ├── preprocessing.py     # Task 1: Data cleaning
│   ├── sentiment_analysis.py # Task 2: Sentiment scoring
│   ├── thematic_analysis.py  # Task 2: Theme extraction
├── tests/               # Unit and sanity tests
├── requirements.txt      # Dependencies
├── final_report.md       # Task 4: Final Business Report
└── README.md
```

---

## How to Run

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/SimretAbebe/fintech-review-analytics.git
cd fintech-review-analytics

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Data Pipeline
Execute the scripts in the following order:
```bash
# Task 1: Collect and Preprocess data
python src/scraper.py
python src/preprocessing.py

# Task 2: Analyze Sentiment and Themes
python src/sentiment_analysis.py
python src/thematic_analysis.py
```

### 3. Database Integration (Task 3)
```bash
python src/database_manager.py
```

### 4. Visualizations & Reporting (Task 4)
Open `notebooks/task4_insights.ipynb` in your preferred Jupyter environment (e.g., VS Code) to generate the final plots. The final business insights are documented in `final_report.md`.

### 5. Running Tests
```bash
# Run sanity tests to verify environment and data
python -m unittest tests/test_pipeline.py
```

---

## Progress Summary
*   **Task 1:** 1,500 reviews collected and preprocessed with 100% data integrity.
*   **Task 2:** Sentiment and Thematic analysis complete with automated visualizations.
*   **Task 3:** PostgreSQL relational schema designed (`scripts/schema.sql`). Automated Python script (`psycopg2`) written to insert and structure the cleaned data. SQL queries written to verify data integrity (`scripts/verify_data.sql`).
*   **Task 4:** Advanced visualizations implemented in Jupyter Notebook (`notebooks/task4_insights.ipynb`)

---

## Technologies Used
*   **Database:** PostgreSQL, `psycopg2`
*   **NLP:** `transformers` (DistilBERT), `spaCy`
*   **Analysis:** `pandas`, `scikit-learn`
*   **Visualization:** `matplotlib`, `seaborn`
*   **CI/CD:** GitHub Actions
