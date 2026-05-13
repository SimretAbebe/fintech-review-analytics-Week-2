import pandas as pd
import os

def preprocess_reviews():
    raw_path = 'data/raw/raw_reviews.csv'
    clean_path = 'data/cleaned_reviews.csv'
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please run the scraper first.")
        return

    # Load the raw data
    df = pd.DataFrame(pd.read_csv(raw_path))
    initial_count = len(df)
    print(f"Starting preprocessing for {initial_count} reviews...")

    # 1. Remove duplicate reviews based on 'reviewId'
    df = df.drop_duplicates(subset=['reviewId'])
    print(f"Removed {initial_count - len(df)} duplicates.")

    # 2. Handle missing values
    # Drop rows missing 'content' (the review text) or 'score' (the rating)
    df = df.dropna(subset=['content', 'score'])
    print(f"Rows remaining after dropping missing values: {len(df)}")

    # 3. Normalize dates to YYYY-MM-DD
    # google-play-scraper usually returns datetime objects, but in CSV they might be strings
    df['at'] = pd.to_datetime(df['at']).dt.strftime('%Y-%m-%d')

    # 4. Select and rename columns as per requirements
    # Required: review, rating, date, bank, source
    clean_df = df[['content', 'score', 'at', 'bank_name', 'source']].copy()
    clean_df.columns = ['review', 'rating', 'date', 'bank', 'source']

    # 5. Save the cleaned dataset
    clean_df.to_csv(clean_path, index=False)
    print(f"\nPreprocessing complete!")
    print(f"Final count: {len(clean_df)} reviews.")
    print(f"Cleaned data saved to: {clean_path}")

if __name__ == "__main__":
    preprocess_reviews()
