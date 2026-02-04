import requests
import base64
import os
import time

# Configuration
API_URL = "https://e50c264b4ed241.lhr.life"  # The PUBLIC deployed URL
API_KEY = "sk_test_123456789"
FILES = [
    ("test_human_voice.mp3", "English"),
    ("test_ai_fake.mp3", "English"),
    ("test_unknown.mp3", "Hindi")
]

def analyze_audio(file_path, language):
    url = f"{API_URL}/api/voice-detection"
    
    # Read and encode audio
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    except FileNotFoundError:
        return {"error": "File not found"}

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
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            data['duration'] = f"{duration:.2f}s"
            return data
        else:
            return {"error": f"Status {response.status_code}", "detail": response.text}
    except Exception as e:
        return {"error": str(e)}

def main():
    print(f"\n🚀 AI VOICE DETECTION BATCH TEST")
    print(f"📡 API Endpoint: {API_URL}")
    print(f"🔑 API Key: {API_KEY[:4]}****")
    print("-" * 80)
    
    results = []
    
    print("Processing...")
    for filename, lang in FILES:
        print(f"   ► Analyzing {filename} ({lang})...", end="\r")
        result = analyze_audio(filename, lang)
        
        if "error" in result:
            results.append([filename, lang, "ERROR", "N/A", result['error']])
        else:
            cls = result.get('classification', 'UNKNOWN')
            conf = result.get('confidenceScore', 0)
            conf_str = f"{conf*100:.2f}%"
            expl = result.get('explanation', '')[:40] + "..."
            results.append([filename, lang, cls, conf_str, expl])
        
        print(f"   ✅ Analyzed {filename}           ")

    print("-" * 80)
    print(f"{'FILENAME':<20} | {'LANG':<10} | {'RESULT':<12} | {'CONFIDENCE':<10} | {'DETAILS'}")
    print("-" * 80)
    
    for row in results:
        print(f"{row[0]:<20} | {row[1]:<10} | {row[2]:<12} | {row[3]:<10} | {row[4]}")
    
    print("-" * 80)
    print("✅ Batch Processing Complete")

if __name__ == "__main__":
    main()
