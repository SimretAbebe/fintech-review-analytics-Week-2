import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_interim_plots():
    input_path = 'data/analyzed_reviews.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load data
    df = pd.read_csv(input_path)
    os.makedirs('data/plots', exist_ok=True)

    # Mean Sentiment by Bank (Bar Chart)
    plt.figure(figsize=(10, 6))
    bank_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values()
    bank_sentiment.plot(kind='barh', color='skyblue')
    plt.title('Mean Positivity Score by Bank')
    plt.xlabel('Positivity Score (0 to 1)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.savefig('data/plots/bank_sentiment.png')
    print("Saved: data/plots/bank_sentiment.png")

    # Theme Distribution (Stacked Bar)
    theme_dist = df.groupby(['bank', 'identified_theme']).size().unstack(fill_value=0)
    theme_dist.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
    plt.title('Review Themes per Bank')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.legend(title='Themes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('data/plots/theme_distribution.png')
    print("Saved: data/plots/theme_distribution.png")
    plt.show()

if __name__ == "__main__":
    generate_interim_plots()
