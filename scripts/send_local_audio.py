import requests
import base64
import sys
from pathlib import Path

API_URL = "http://localhost:8000"
API_KEY = "sk_test_123456789"

def send(file_path: str, language: str = "English"):
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {file_path}")
        return
    with open(p, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64,
    }
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    r = requests.post(f"{API_URL}/api/voice-detection", json=payload, headers=headers)
    print(f"Status: {r.status_code}")
    print(r.json())

if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "myvoice.mp3"
    lang_arg = sys.argv[2] if len(sys.argv) > 2 else "English"
    send(file_arg, lang_arg)
