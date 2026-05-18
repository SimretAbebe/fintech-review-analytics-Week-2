import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise

def preprocess_text(text: str) -> str:
    """
    Tokenize, remove stop words, and lemmatize text.
    Lemmatization is used to group word variations (e.g., 'crashed' -> 'crash')
    for better theme frequency analysis.
    """
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def get_theme(text: str) -> str:
    """
    Categorize review based on business-relevant keywords.
    Logic: We prioritize specific technical blockers (Account Access, Transactions) 
    over general feedback as these are the most actionable for product teams.
    """
    text = str(text).lower()
    
    themes = {
        "Account Access": ["login", "signin", "password", "otp", "code", "access", "account", "register"],
        "Transaction Performance": ["transfer", "transaction", "payment", "slow", "speed", "failed", "pending", "money", "sent"],
        "UI & Design": ["ui", "interface", "design", "look", "easy", "clean", "beautiful", "navigation"],
        "Customer Support": ["support", "call", "service", "help", "agent", "response", "customer"]
    }
    
    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            return theme
    return "Feature Request / General"

def run_thematic_analysis():
    """Main function to run the thematic extraction pipeline."""
    input_path = 'data/sentiment_reviews.csv'
    output_path = 'data/analyzed_reviews.csv'
    
    if not os.path.exists(input_path):
        logging.error(f"Input file {input_path} missing. Ensure Task 2 (Sentiment) has been run.")
        return

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Starting thematic analysis on {len(df)} reviews...")

       
        df['clean_text'] = df['review'].apply(preprocess_text)

       
        df['identified_theme'] = df['review'].apply(get_theme)


        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        vectorizer.fit_transform(df['clean_text'])
        top_keywords = vectorizer.get_feature_names_out()
        logging.info(f"Top keywords identified: {', '.join(top_keywords)}")

      
        # Save with clean column structure, ensuring all necessary DB columns are preserved
        result_df = df[['review', 'rating', 'date', 'bank', 'sentiment_label', 'sentiment_score', 'identified_theme']]
        result_df.to_csv(output_path, index_label='review_id')
        logging.info(f"Thematic analysis complete. Data saved to {output_path}")

    except Exception as e:
        logging.error(f"Unexpected error during thematic analysis: {e}")

if __name__ == "__main__":
    run_thematic_analysis()
