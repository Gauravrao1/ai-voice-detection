"""
Complete API testing script
"""

import requests
import base64
import json
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path to import from src if needed
sys.path.append(str(Path(__file__).parent.parent))

API_URL = "http://localhost:8000"
API_KEY = "sk_test_123456789"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        logger.info(f"Health Check: {response.json()}")
        if response.status_code == 200:
            print("✅ Health Check Passed")
        else:
            print("❌ Health Check Failed")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")

def test_voice_detection(audio_file_path, language="English"):
    """Test voice detection with real audio"""
    if not Path(audio_file_path).exists():
        logger.warning(f"File not found: {audio_file_path}")
        return

    # Read and encode audio
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # API request
    payload = {
        "language": language,
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
        
        print(f"\nDetection Result for {Path(audio_file_path).name}:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✅ Detection Request Passed")
        else:
            print("❌ Detection Request Failed")
            
    except Exception as e:
        print(f"❌ Request Failed: {e}")

def run_all_tests():
    """Run all tests"""
    print("🧪 Starting API Tests...")
    print(f"Target URL: {API_URL}")
    print(f"API Key: {API_KEY}")
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Check for sample file
    sample_file = Path("sample_voice_1.mp3")
    if not sample_file.exists():
        # Create a dummy file if not exists just for structure (won't work for real detection)
        # Or look for other samples
        print("\n⚠️ No sample file 'sample_voice_1.mp3' found in root.")
        print("Please place an MP3 file to test detection.")
    else:
        test_voice_detection(sample_file, language="English")
    
    print("\n✅ Test execution complete!")

if __name__ == "__main__":
    run_all_tests()
