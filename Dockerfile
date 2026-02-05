FROM python:3.12-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Create necessary directories with proper permissions
RUN mkdir -p ml_model/saved_models models/huggingface_cache temp && \
    chmod -R 755 ml_model models temp

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models/huggingface_cache

# Expose port
EXPOSE 8000

# Run the startup script
CMD ["./start.sh"]