import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME", "bank_reviews"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "password123"),  
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432")
}

def create_database_if_not_exists():
    """Create the bank_reviews database if it doesn't already exist."""
    try:
        default_config = DB_CONFIG.copy()
        default_config["dbname"] = "postgres"
        
        conn = psycopg2.connect(**default_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM pg_database WHERE datname='bank_reviews'")
        if not cur.fetchone():
            logging.info("Database 'bank_reviews' does not exist. Creating it automatically...")
            cur.execute("CREATE DATABASE bank_reviews")
            logging.info("Database 'bank_reviews' created successfully.")
            
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to create database: {e}")
        raise

def init_db():
    """Create tables using the schema.sql file."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        with open('scripts/schema.sql', 'r') as f:
            cur.execute(f.read())
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info("Database schema initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

def insert_data():
    """Insert data from analyzed_reviews.csv into PostgreSQL."""
    input_path = 'data/analyzed_reviews.csv'
    if not os.path.exists(input_path):
        logging.error(f"{input_path} not found. Run Task 1 and 2 pipelines first.")
        return

    try:
        df = pd.read_csv(input_path)
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Get bank_id mapping from the metadata table
        cur.execute("SELECT bank_id, bank_name FROM banks")
        bank_map = {name: b_id for b_id, name in cur.fetchall()}

        logging.info(f"Inserting {len(df)} reviews into the database...")
        inserted_count = 0

        for _, row in df.iterrows():
            bank_id = bank_map.get(row['bank'])
            if not bank_id:
                continue
            
            cur.execute("""
                INSERT INTO reviews (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, identified_theme)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                bank_id, 
                row['review'], 
                int(row['rating']), 
                row['date'], 
                row['sentiment_label'], 
                float(row['sentiment_score']), 
                row['identified_theme']
            ))
            inserted_count += 1

        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"Data insertion complete! Successfully inserted {inserted_count} records.")

    except Exception as e:
        logging.error(f"An error occurred during data insertion: {e}")

if __name__ == "__main__":
    logging.info("Starting Task 3: Database Engineering Pipeline")
    try:
        create_database_if_not_exists()
        init_db()
        insert_data()
    except Exception as e:
        logging.error("Pipeline failed. Ensure PostgreSQL is running.")
