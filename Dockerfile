# Use a lightweight but ML-compatible Python image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Reduce memory spikes from tokenizers
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies required for FAISS & PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Start the Flask app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "--timeout", "120"]
