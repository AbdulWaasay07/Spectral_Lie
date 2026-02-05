# Spectral Lie: AI Voice Detection System

A professional-grade machine learning pipeline and API for distinguish between human speech and AI-generated (synthetic) audio.

## 🚀 Quick Start (For Judges & Testers)

The easiest way to run the entire system is using **Docker**. This ensures all dependencies (Redis, ML models, feature extractors) are perfectly configured.

### 1. Launch the Server
Navigate to the API directory and start the containers:
```powershell
cd "part3_api"
docker compose up --build -d
```
*Wait ~30 seconds for the containers to initialize and the ML model to load.*

### 2. Run a Test Case
Use the provided tester script to send human audio to the API:
```powershell
python run_test.py
```
**View Results**: Open `test_result.json` in this directory to see the classification, calibrated confidence score, and the AI's explanation.

---

## 🛠 Project Architecture

The project is divided into three specialized modules:

### Part 1: Audio Feature Engineering (`/part1_audio_features`)
Extracts deep spectral and acoustic characteristics from raw audio.
*   **Acoustic**: Jitter, Shimmer, HNR, Pitch variance, and 13-band MFCCs.
*   **Deep**: 1536-dimensional deep embeddings for complex pattern recognition.
*   **Robustness**: Handles format conversion (MP3/WAV) and noise filtering.

### Part 2: Detection & Calibration (`/part2_detection`)
The "Brain" of the system.
*   **Model**: A neural network classifier optimized for acoustic nuances.
*   **Calibration**: Uses **Platt Scaling** and **Label Smoothing** to ensure confidence scores are realistic (0.4 - 0.9) rather than overconfident 0s and 1s.
*   **Explainability**: Generates human-readable reasons for every verdict based on melodic stability and spectral cues.

### Part 3: Production API (`/part3_api`)
Enterprise-ready deployment layer.
*   **UI/API**: FastAPI-based REST endpoints.
*   **Security**: API Key authentication (`X-API-Key`).
*   **Performance**: Redis-backed caching and asynchronous processing.

---

## 💻 Local Development Setup (Manual)

If you prefer to run without Docker:

1.  **Dependencies**: Install requirements in each folder:
    ```powershell
    pip install -r part1_audio_features/requirements.txt
    pip install -r part2_detection/requirements.txt
    pip install -r part3_api/requirements.txt
    ```
2.  **Redis**: Ensure a Redis server is running on `localhost:6379`.
3.  **Start API**:
    ```powershell
    cd part3_api
    uvicorn app.main:app --port 8000
    ```

## 📊 Evaluation Metrics
The model is evaluated on its ability to detect "Deepfake" voices across multiple languages, focusing on:
*   **Detection Accuracy**: High AUC on synthetic-human overlap.
*   **Probability Calibration**: Reliability of the confidence score.
*   **Latency**: Real-time processing capability for short audio bursts.
