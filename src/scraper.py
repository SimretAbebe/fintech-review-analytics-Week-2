from google_play_scraper import Sort, reviews
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def scrape_bank_reviews():
    """
    Scrape reviews for CBE, BOA, and Dashen Bank.
    Strategy: Target 500 reviews per bank to ensure we meet the 1,200+ total 
    requirement for the project while staying within rate limits.
    """
    apps = {
        'Commercial Bank of Ethiopia': 'com.cbe.cbebirr',
        'Bank of Abyssinia': 'com.boa.banking',
        'Dashen Bank': 'com.dashen.amole'
    }

    all_reviews = []
    raw_dir = 'data/raw'
    
 
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        logging.info(f"Created directory: {raw_dir}")

    for bank_name, app_id in apps.items():
        logging.info(f"Scraping reviews for {bank_name}...")
        try:
            result, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=500
            )
            for r in result:
                r['bank'] = bank_name
            all_reviews.extend(result)
            logging.info(f"Successfully collected {len(result)} reviews for {bank_name}.")
        except Exception as e:
            logging.error(f"Failed to scrape data for {bank_name} ({app_id}): {e}")

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        output_file = os.path.join(raw_dir, 'raw_reviews.csv')
        df.to_csv(output_file, index=False)
        logging.info(f"Scraping complete! Total reviews collected: {len(all_reviews)}")
        logging.info(f"Raw data saved to: {output_file}")
    else:
        logging.warning("No reviews were collected. Check network connection or app IDs.")

if __name__ == "__main__":
    scrape_bank_reviews()
