import requests
import base64
from pathlib import Path

API_URL = "https://e50c264b4ed241.lhr.life"
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
    r = requests.post(f"{API_URL}/api/voice-detection", json=payload, headers=headers, verify=False)
    print(f"Status: {r.status_code}")
    try:
        print(r.json())
    except Exception:
        print(r.text)

if __name__ == "__main__":
    send("myvoice.mp3", "English")
