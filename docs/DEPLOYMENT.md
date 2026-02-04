# Deployment Guide

## Option 1: Render.com (Easiest)

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect GitHub repository
4. Add environment variables
5. Deploy!

## Option 2: Docker
```bash
# Build
docker build -t ai-voice-detection .

# Run
docker run -p 8000:8000 \
  -e API_KEY=your_key \
  -e MODEL_TYPE=hybrid \
  ai-voice-detection
```

## Option 3: Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export API_KEY=your_key
export MODEL_TYPE=hybrid

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## GPU Setup (Optional)

For better performance, use GPU:
```bash
# Install CUDA toolkit
# Set environment
export USE_GPU=true

# Run
python -m app.main
```