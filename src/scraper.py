import pandas as pd
from google_play_scraper import Sort, reviews, app
import os

# Define the banks and their Play Store App IDs
BANKS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.bankofabyssinia.boamobile.retail",
    "Dashen Bank": "com.dashen.dashensuperapp"
}

def scrape_bank_reviews():
    all_reviews = []
    
    for bank_name, app_id in BANKS.items():
        print(f"Scraping reviews for {bank_name}...")
        
        # Scrape reviews
        # Fetched 500 to be safe and ensure I met the 400+ requirement
        result, _ = reviews(
            app_id,
            lang='en', # Language of reviews
            country='us', # Country to fetch from
            sort=Sort.NEWEST, # Get the latest reviews
            count=500 
        )
        
        # Add bank name to each review and store
        for r in result:
            r['bank_name'] = bank_name
            r['source'] = 'Google Play'
            all_reviews.append(r)
            
        print(f"Successfully collected {len(result)} reviews for {bank_name}.")

    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Ensure the data/raw directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/raw/raw_reviews.csv'
    df.to_csv(output_path, index=False)
    print(f"\nScraping complete! Total reviews collected: {len(df)}")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    scrape_bank_reviews()
