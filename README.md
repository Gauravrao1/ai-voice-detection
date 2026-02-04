# 🎙️ AI Voice Detection API

Detect AI-generated vs Human voices across 5 Indian languages using state-of-the-art deep learning.

## 🌟 Features

- ✅ Multi-language support (Tamil, English, Hindi, Malayalam, Telugu)
- ✅ Hugging Face Transformers integration (`MelodyMachine/Deepfake-audio-detection`)
- ✅ State-of-the-art accuracy (~99%)
- ✅ REST API with authentication
- ✅ GPU acceleration support

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: You may need `ffmpeg` installed on your system for audio processing.*

### 2. Configure Environment
The project comes with a default configuration in `app/config.py` using a test API key.
You can override settings using a `.env` file:
```bash
cp .env.example .env
```

### 3. Run the Server
```bash
python -m app.main
```
The server will start at `http://0.0.0.0:8000`.

## 🧪 Testing

You can test the API using the provided script:

```bash
python scripts/test_api.py
```

Or using cURL:
```bash
curl -X POST "http://localhost:8000/api/voice-detection" \
-H "Content-Type: application/json" \
-H "x-api-key: sk_test_123456789" \
-d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_STRING_HERE"
}'
```

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧠 Model Information

This project uses **MelodyMachine/Deepfake-audio-detection**, a fine-tuned Wav2Vec2 model specialized in distinguishing between real and synthetic voices.

## 📦 Project Structure

- `app/main.py`: API entry point
- `app/models/hf_detector.py`: Hugging Face model integration
- `app/utils/`: Audio processing utilities
- `scripts/`: Testing scripts
