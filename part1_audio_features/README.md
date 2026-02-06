# Part 1: Audio Ingestion & Feature Engineering Module

## Purpose
This module handles the ingestion of raw audio (Base64 encoded MP3), validates it, preprocesses it, and extracts both acoustic and deep learning features.

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   
   ```

3. Ensure `ffmpeg` is installed on your system.

## Usage

```python
from part1 import extract_features

# data = "base64_encoded_mp3_string..."
# bundle = extract_features(data, language_hint='ta')
# print(bundle.acoustic_features)
```

## Features
- **Acoustic**: MFCC, Pitch (F0), Jitter, Shimmer, HNR, Spectral stats.
- **Deep**: Wav2Vec2-base embeddings (mean pooled).
- **Metadata**: Duration, sample rate, hash.

## Testing
Run unit tests:
```bash
pytest tests
```
