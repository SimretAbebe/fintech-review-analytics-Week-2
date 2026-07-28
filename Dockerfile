# image smaller and faster to build/deploy.
FROM python:3.11-slim

# All following commands run inside this folder within the container.
WORKDIR /app

# Copy ONLY the requirements file first, before the rest of the code.

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Now copy the actual application code and the pre-built artifacts it
# depends on (the trained model, the processed dataset, the feature
# column list, and the Streamlit theme config).
COPY app.py .
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY data/labeled_reviews_multilingual_fixed.csv ./data/labeled_reviews_multilingual_fixed.csv
COPY data/feature_columns.json ./data/feature_columns.json
COPY models/risk_model.joblib ./models/risk_model.joblib

# Documents which network port the container listens on. This alone

EXPOSE 8501

# Start the dashboard. Two settings matter specifically because we're
# inside a container:
#   --server.address=0.0.0.0  makes Streamlit listen on every network
#     interface, not just "localhost" - without this, nothing outside
#     the container could reach it, even with the port published.
#   --server.headless=true    stops Streamlit from trying to open a
#     browser window, which doesn't exist inside a container anyway.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]