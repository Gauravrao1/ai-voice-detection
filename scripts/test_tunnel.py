import requests
import base64
import json
import sys
from pathlib import Path

# The public URL from the tunnel
API_URL = "https://b5ac0dd7592d5c.lhr.life"
API_KEY = "sk_test_123456789"

def test_public_access():
    print(f"🌍 Testing Public URL: {API_URL}")
    
    # 1. Test Root/Health
    try:
        print("1. Checking Root Endpoint...")
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ Public Root Accessible")
            print(f"   Response: {response.json()['service']}")
        else:
            print(f"❌ Root Check Failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 2. Test Detection
    print("\n2. Checking Voice Detection Endpoint (POST)...")
    sample_file = Path("sample_voice_1.mp3")
    if not sample_file.exists():
        print("⚠️ Sample file missing, skipping detection test")
        return

    with open(sample_file, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Public Detection Successful!")
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidenceScore']}")
        else:
            print(f"❌ Detection Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Request Error: {e}")

if __name__ == "__main__":
    test_public_access()
