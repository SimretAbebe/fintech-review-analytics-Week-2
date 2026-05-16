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
├── src/
│   ├── scraper.py           # Task 1: Data collection
│   ├── preprocessing.py     # Task 1: Data cleaning
│   ├── sentiment_analysis.py # Task 2: NLP scoring
│   ├── thematic_analysis.py  # Task 2: Theme extraction
│   └── visualizations.py     # Task 2: Interim plots
├── requirements.txt      # Dependencies
└── README.md
```

---

## Progress Summary

### Task 1: Data Collection and Preprocessing
*   **Scraping:** Successfully collected **1,500 reviews** (500 per bank) using `google-play-scraper`.
*   **Preprocessing:** Cleaned raw data by removing duplicates, handling missing values, and normalizing dates to `YYYY-MM-DD`.
*   **KPIs:** Met the 1,200+ review threshold with 100% data integrity.

### Task 2: Sentiment and Thematic Analysis
*   **Sentiment Analysis:** Utilized the `DistilBERT` transformer model to assign positivity scores (0.0 to 1.0).
*   **Thematic Analysis:** Used `spaCy` to categorize reviews into 5 business themes: *Account Access, Transaction Performance, UI & Design, Customer Support, and Feature Requests.*
*   **Visualizations:** Generated comparative charts showing sentiment distribution and theme frequency per bank.
*   **Key Finding:** Identified **Account Access** (login/OTP issues) as the primary driver of negative sentiment for Bank of Abyssinia (BOA).

---

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SimretAbebe/fintech-review-analytics.git
   cd fintech-review-analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Run the pipeline:**
   ```bash
   # 1. Scrape data
   python src/scraper.py
   # 2. Clean data
   python src/preprocessing.py
   # 3. Analyze sentiment
   python src/sentiment_analysis.py
   # 4. Extract themes
   python src/thematic_analysis.py
   # 5. Generate plots
   python src/visualizations.py
   ```

---

## Technologies Used
*   **Languages:** Python 3.11
*   **NLP:** `transformers` (DistilBERT), `spaCy`
*   **Data Science:** `pandas`, `numpy`, `scikit-learn`
*   **Visualization:** `matplotlib`, `seaborn`
*   **CI/CD:** GitHub Actions

