import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """Tokenize, remove stop words, and lemmatize."""
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def get_theme(text):
    """Categorize review based on keywords."""
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
    input_path = 'data/sentiment_reviews.csv'
    output_path = 'data/analyzed_reviews.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Wait for sentiment analysis to finish.")
        return

    df = pd.read_csv(input_path)
    print(f"Starting thematic analysis on {len(df)} reviews...")

    
    print("Cleaning text (lemmatization and stop-word removal)...")
    df['clean_text'] = df['review'].apply(preprocess_text)

    
    print("Mapping reviews to business themes...")
    df['identified_theme'] = df['review'].apply(get_theme)

    
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    top_keywords = vectorizer.get_feature_names_out()
    print(f"\nTop keywords across all reviews: {', '.join(top_keywords)}")

    
    final_df = df.copy()
    
    final_df['review_id'] = final_df.index
    
    # Reorder columns as per requirements
    result_df = final_df[['review_id', 'review', 'bank', 'sentiment_label', 'sentiment_score', 'identified_theme']]
    
    
    result_df.to_csv(output_path, index=False)
    print(f"\nThematic analysis complete! Final data saved to {output_path}")
    
    # Show theme distribution
    print("\nTheme Distribution per Bank:")
    print(df.groupby(['bank', 'identified_theme']).size().unstack(fill_value=0))

if __name__ == "__main__":
    run_thematic_analysis()
