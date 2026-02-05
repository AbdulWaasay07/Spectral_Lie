# How to Run the Detection System

I have created a helper script to make this easy.

1. **Ensure the Server is Running**
   The system runs in Docker. It should be up, but if you need to restart it:
   ```bash
   docker compose up -d
   ```

2. **Run the Test Script**
   I created a Python script that reads your `test_audio.b64` and sends it to the API.
   ```bash
   python run_test.py
   ```

3. **Check Results**
   The script will print the Classification (Human/AI) and Confidence score.
   It also saves the full result to `test_result.json`.
