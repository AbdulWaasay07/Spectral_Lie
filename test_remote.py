
import requests
import json
import time

BASE_URL = "https://spectral-lie.onrender.com"
HEADERS = {"x-api-key": "test-key-123"}

print(f"Checking health at {BASE_URL}/ready...")
try:
    resp = requests.get(f"{BASE_URL}/ready", headers=HEADERS, timeout=10)
    print(f"Health Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"❌ Health check failed: {e}")
    exit(1)

print("\nSending detection request...")
try:
    try:
        with open("part3_api/test_audio.b64", "r", encoding="utf-8") as f:
            audio_b64 = f.read().strip()
    except UnicodeDecodeError:
        with open("part3_api/test_audio.b64", "r", encoding="utf-16") as f:
            audio_b64 = f.read().strip()
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
    
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/detect-voice", headers=HEADERS, json=payload, timeout=60)
    duration = time.time() - t0
    
    print(f"Status: {resp.status_code}", flush=True)
    print(f"Time: {duration:.2f}s", flush=True)
    print(f"Response: {resp.text[:500]}", flush=True)

except Exception as e:
    print(f"❌ Request failed: {e}", flush=True)

print("Script finished.", flush=True)
