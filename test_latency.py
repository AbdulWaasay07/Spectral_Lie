# test_latency.py - Local latency measurement
import time
import requests
import json

# Read the base64 audio from test file (handle UTF-16 encoding)
try:
    with open("part3_api/test_audio.b64", "r", encoding="utf-8") as f:
        audio_b64 = f.read().strip()
except UnicodeDecodeError:
    with open("part3_api/test_audio.b64", "r", encoding="utf-16") as f:
        audio_b64 = f.read().strip()

print(f"Audio payload size: {len(audio_b64)} bytes")
print("=" * 60)
print("Starting API request...")
print("=" * 60)

t0 = time.time()

try:
    resp = requests.post(
        "http://localhost:8000/detect-voice",
        headers={"x-api-key": "test-key-123"},
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": audio_b64
        },
        timeout=15  # Give it 15 seconds max
    )
    
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print(f"✓ Request completed")
    print("=" * 60)
    print(f"Status: {resp.status_code}")
    print(f"Time: {elapsed:.2f} seconds")
    print()
    
    # Interpret timing
    if elapsed < 3:
        print("✅ EXCELLENT - Safe for all environments")
    elif elapsed < 6:
        print("⚠️  BORDERLINE - Should work but optimize if possible")
    elif elapsed < 10:
        print("❌ RISKY - Hackathon timeout risk")
    else:
        print("❌ FAILED - Will timeout in hackathon")
    
    print("\nResponse:")
    print(json.dumps(resp.json(), indent=2))
    
except requests.exceptions.Timeout:
    elapsed = time.time() - t0
    print(f"\n❌ TIMEOUT after {elapsed:.2f}s - API too slow!")
    
except requests.exceptions.ConnectionError:
    print("\n❌ CONNECTION ERROR - Is the server running?")
    print("Run: uvicorn part3_api.app.main:app --reload --port 8000")
    
except Exception as e:
    elapsed = time.time() - t0
    print(f"\n❌ ERROR after {elapsed:.2f}s: {e}")
