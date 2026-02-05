import requests
import json
import base64

API_URL = "http://localhost:8000/detect-voice"
API_KEY = "test-key-123"

# Tiny invalid base64 audio to test connectivity/headers
payload = {
    "audio_base64": "SUQzBAAAAAAAI1REU4AAAAAA", # Dummy ID3 tag start
    "language": "English"
}

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

try:
    print("Sending dummy request...")
    response = requests.post(API_URL, json=payload, headers=headers)
    print(response.status_code)
    print(response.text)
except Exception as e:
    print(e)
