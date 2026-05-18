-- Verify Data Integrity

-- 1. Count total reviews per bank
SELECT 
    b.bank_name, 
    COUNT(r.review_id) as total_reviews
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY total_reviews DESC;

-- 2. Compute average rating per bank
SELECT 
    b.bank_name, 
    ROUND(AVG(r.rating), 2) as average_rating
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY average_rating DESC;


SELECT 
    COUNT(*) as null_review_text_count
FROM reviews 
WHERE review_text IS NULL;

SELECT 
    COUNT(*) as null_sentiment_score_count
FROM reviews 
WHERE sentiment_score IS NULL;

-- 4. Count themes extracted (to verify NLP pipeline integration)
SELECT 
    identified_theme, 
    COUNT(*) as theme_count
FROM reviews
GROUP BY identified_theme
ORDER BY theme_count DESC;
