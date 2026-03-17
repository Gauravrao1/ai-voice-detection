# 🎙️ AI Voice Detection API

Detect AI-generated vs Human voices across multiple Indian languages using deep learning.

--------------------------------------------------

FEATURES

- Multi-language support (Tamil, English, Hindi, Malayalam, Telugu)
- Hugging Face Transformers integration
- Model: MelodyMachine/Deepfake-audio-detection
- High accuracy (~99%)
- REST API with API key authentication
- GPU acceleration support

--------------------------------------------------

QUICK START

1. Install Dependencies

pip install -r requirements.txt

Note: Install ffmpeg for audio processing.

--------------------------------------------------

2. Configure Environment

cp .env.example .env

Edit the .env file if needed.

--------------------------------------------------

3. Run the Server

python -m app.main

Server will start at:
http://0.0.0.0:8000

--------------------------------------------------

TESTING

Using Python script:

python scripts/test_api.py

--------------------------------------------------

Using cURL:

curl -X POST "http://localhost:8000/api/voice-detection" \
-H "Content-Type: application/json" \
-H "x-api-key: sk_test_123456789" \
-d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_STRING_HERE"
}'

--------------------------------------------------

API DOCUMENTATION

Swagger UI:
http://localhost:8000/docs

ReDoc:
http://localhost:8000/redoc

--------------------------------------------------

MODEL INFORMATION

Model: MelodyMachine/Deepfake-audio-detection

- Based on Wav2Vec2
- Fine-tuned for deepfake audio detection
- Supports multilingual classification

--------------------------------------------------

PROJECT STRUCTURE

app/
  main.py
  config.py
  models/
    hf_detector.py
  utils/

scripts/
  test_api.py

requirements.txt
.env.example
README.md

--------------------------------------------------

AUTHENTICATION

All requests require API key:

x-api-key: sk_test_123456789

You can change it in:
app/config.py
or via .env

--------------------------------------------------

PERFORMANCE TIPS

- Use GPU for faster inference
- Use WAV format for better accuracy
- Keep audio length between 5–15 seconds

--------------------------------------------------

USE CASES

- Deepfake voice detection
- Fraud and scam prevention
- Cybersecurity systems
- Media verification

--------------------------------------------------

CONTRIBUTING

1. Fork the repo
2. Create a branch
3. Commit changes
4. Open a pull request

--------------------------------------------------

LICENSE

MIT License

--------------------------------------------------

SUPPORT

Star the repository if you find it useful.
