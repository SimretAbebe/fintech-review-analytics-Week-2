import pandas as pd
from transformers import pipeline
import os

def run_sentiment_analysis():
    input_path = 'data/cleaned_reviews.csv'
    output_path = 'data/sentiment_reviews.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reviews for sentiment analysis...")

    # Initialize the pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    results = []
    print("Scoring sentiment (with positivity normalization)...")
    for i, text in enumerate(df['review']):
        try:
            result = sentiment_pipeline(str(text)[:512])[0]
            label = result['label']
            confidence = result['score']
            
          
            positivity_score = confidence if label == 'POSITIVE' else (1 - confidence)
    
            if 0.45 <= positivity_score <= 0.55:
                final_label = 'NEUTRAL'
            elif positivity_score > 0.55:
                final_label = 'POSITIVE'
            else:
                final_label = 'NEGATIVE'

            results.append({
                'label': final_label,
                'score': positivity_score
            })
        except Exception:
            results.append({'label': 'NEUTRAL', 'score': 0.5})
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} reviews...")

    # Update dataframe
    results_df = pd.DataFrame(results)
    df['sentiment_label'] = results_df['label']
    df['sentiment_score'] = results_df['score']

    df.to_csv(output_path, index=False)
    print(f"Analysis complete! Saved to {output_path}")

  
    print("\n" + "="*40)
    print("FINAL TASK 2 RESULTS (NORMALIZED)")
    print("="*40)
    print("\nMean Positivity Score by Bank (0=Neg, 1=Pos):")
    print(df.groupby('bank')['sentiment_score'].mean().sort_values(ascending=False))
    
    print("\nMean Positivity Score by Star Rating:")
    print(df.groupby('rating')['sentiment_score'].mean())
    print("="*40)

if __name__ == "__main__":
    run_sentiment_analysis()
