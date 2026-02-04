"""
Complete API testing script
"""

import requests
import base64
import json
from pathlib import Path
from loguru import logger

API_URL = "http://localhost:8000"
API_KEY = "sk_test_123456789"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    logger.info(f"Health Check: {response.json()}")
    assert response.status_code == 200

def test_voice_detection(audio_file_path, language="English"):
    """Test voice detection with real audio"""
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
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        json=payload,
        headers=headers
    )
    
    logger.info(f"Detection Result for {audio_file_path}:")
    logger.info(json.dumps(response.json(), indent=2))
    
    assert response.status_code == 200
    return response.json()

def test_invalid_api_key():
    """Test with invalid API key"""
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "fake_base64_data"
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_key"
    }
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        json=payload,
        headers=headers
    )
    
    logger.info(f"Invalid API Key Test: {response.status_code}")
    assert response.status_code == 401

def run_all_tests():
    """Run all tests"""
    logger.info("🧪 Starting API Tests...")
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Invalid API key
    test_invalid_api_key()
    
    # Test 3: Real audio files (if available)
    test_audio_dir = Path("tests/test_audio_samples")
    if test_audio_dir.exists():
        for audio_file in test_audio_dir.glob("*.mp3"):
            test_voice_detection(audio_file)
    
    logger.info("✅ All tests passed!")

if __name__ == "__main__":
    run_all_tests()