import requests
import json
import os

# Configuration
API_URL = "http://localhost:8000/detect-voice"
API_KEY = "test-key-123"
AUDIO_FILE = "test_audio.b64"

def run_test():
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found.")
        return

    print(f"Reading {AUDIO_FILE}...")
    try:
        # Try reading as binary to detect encoding or just filter bytes
        with open(AUDIO_FILE, "rb") as f:
            raw_bytes = f.read()
        
        # 1. Attempt to decode common text encodings or just treat as ascii
        # Windows PowerShell often creates UTF-16LE files with BOM
        if raw_bytes.startswith(b'\xff\xfe'):
            content = raw_bytes.decode("utf-16le")
        elif raw_bytes.startswith(b'\xfe\xff'):
            content = raw_bytes.decode("utf-16be")
        else:
            # Try utf-8, fallback to latin-1
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("latin-1")

        # 2. Filter to keep ONLY valid Base64 characters (A-Z, a-z, 0-9, +, /, =)
        # This removes newlines, BOM artifacts, spaces, etc.
        import re
        audio_b64 = re.sub(r'[^A-Za-z0-9+/=]', '', content)

    except Exception as e:
        print(f"Error reading file: {e}")
        return
            
    # Basic validation
    if not audio_b64:
        print("Error: Audio file is empty.")
        return

    payload = {
        "audio_base64": audio_b64,
        "language": "English"
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print(f"Status Mapping: {response.status_code}")
        if response.status_code == 200:
            print("\n--- Detection Result (NEW) ---")
            data = response.json()
            print(f"Classification: {data.get('classification')}")
            print(f"Confidence:     {data.get('confidence')}")
            print(f"Explanation:    {data.get('explanation')}")
            
            # Save output for review
            with open("test_result.json", "w") as f:
                # Add a local timestamp to verify update
                import datetime
                data["_test_run_at"] = datetime.datetime.now().isoformat()
                json.dump(data, f, indent=2)
            print(f"\n[UPDATED] Full result saved to {os.path.abspath('test_result.json')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    run_test()
