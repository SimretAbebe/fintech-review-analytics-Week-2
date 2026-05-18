-- Create Banks Table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL,
    app_name VARCHAR(255)
);

-- Create Reviews Table
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id),
    review_text TEXT,
    rating INTEGER,
    review_date DATE,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    identified_theme VARCHAR(50),
    source VARCHAR(50) DEFAULT 'Google Play'
);

-- Insert Initial Bank Metadata
INSERT INTO banks (bank_name, app_name) VALUES 
('Commercial Bank of Ethiopia', 'CBE Birr'),
('Bank of Abyssinia', 'BOA Mobile Banking'),
('Dashen Bank', 'Amole')
ON CONFLICT (bank_name) DO NOTHING;
