import pandas as pd
from transformers import pipeline
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_sentiment_analysis():
    """
    Perform sentiment scoring using DistilBERT.
    Logic: Converts binary model output (Positive/Negative) into a continuous 
    score to allow for fine-grained analysis.
    """
    input_path = 'data/cleaned_reviews.csv'
    output_path = 'data/sentiment_reviews.csv'
    
    if not os.path.exists(input_path):
        logging.error(f"Input file {input_path} missing. Run preprocessing first.")
        return

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded {len(df)} reviews for sentiment analysis.")

        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        results = []
        for i, text in enumerate(df['review']):
            try:
                
                result = sentiment_pipeline(str(text)[:512])[0]
                label = result['label']
                conf = result['score']
                
               
                pos_score = conf if label == 'POSITIVE' else (1 - conf)
                
              
                if 0.45 <= pos_score <= 0.55:
                    final_label = 'NEUTRAL'
                elif pos_score > 0.55:
                    final_label = 'POSITIVE'
                else:
                    final_label = 'NEGATIVE'

                results.append({'label': final_label, 'score': pos_score})
            except Exception as e:
                logging.warning(f"Failed to score review index {i}: {e}")
                results.append({'label': 'NEUTRAL', 'score': 0.5})

            if (i + 1) % 200 == 0:
                logging.info(f"Progress: {i + 1}/{len(df)} reviews processed.")

        
        res_df = pd.DataFrame(results)
        df['sentiment_label'] = res_df['label']
        df['sentiment_score'] = res_df['score']

        df.to_csv(output_path, index=False)
        logging.info(f"Sentiment analysis complete. Results saved to {output_path}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_sentiment_analysis()
