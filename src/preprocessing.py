import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_reviews(input_path: str, output_path: str):
    """
    Clean the raw review dataset.
    Steps:
    1. Drop duplicates by review ID.
    2. Remove entries with missing text or ratings.
    3. Normalize date formats for time-series analysis.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file {input_path} missing. Run the scraper first.")
        return

    try:
        df = pd.read_csv(input_path)
        initial_count = len(df)
        logging.info(f"Loaded {initial_count} reviews for preprocessing.")

       
        if 'reviewId' in df.columns:
            df = df.drop_duplicates(subset=['reviewId'])
            logging.info(f"Removed {initial_count - len(df)} duplicate reviews.")

        df = df.dropna(subset=['content', 'score'])
        logging.info(f"Rows remaining after dropping missing values: {len(df)}")

       
        df['at'] = pd.to_datetime(df['at']).dt.strftime('%Y-%m-%d')

       
        df = df.rename(columns={'content': 'review', 'score': 'rating', 'at': 'date'})
        
        final_df = df[['review', 'rating', 'date', 'bank']]
        final_df.to_csv(output_path, index=False)
        logging.info(f"Preprocessing complete. Cleaned data saved to {output_path}")

    except Exception as e:
        logging.error(f"Error during preprocessing pipeline: {e}")

if __name__ == "__main__":
    clean_reviews('data/raw/raw_reviews.csv', 'data/cleaned_reviews.csv')
