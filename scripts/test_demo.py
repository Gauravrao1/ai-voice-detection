import requests

API_URL = "http://localhost:8000"

def test_demo():
    try:
        print("Checking /demo endpoint...")
        response = requests.get(f"{API_URL}/demo")
        if response.status_code == 200:
            if "<!DOCTYPE html>" in response.text:
                print("✅ /demo is serving HTML")
            else:
                print("❌ /demo returned 200 but content doesn't look like HTML")
                print(response.text[:100])
        else:
            print(f"❌ /demo failed with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    test_demo()
