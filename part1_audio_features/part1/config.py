import os

# Audio constraints
SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 30.0
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(os.path.dirname(BASE_DIR), "temp_audio")

# Feature Extraction
# CRITICAL: Default to False to avoid loading heavy wav2vec2 model
_env_value = os.getenv("USE_DEEP_FEATURES", "false").lower()
USE_DEEP_FEATURES = _env_value in ("true", "1", "yes")

# Debug log
print(f"[part1/config] USE_DEEP_FEATURES env='{os.getenv('USE_DEEP_FEATURES', 'NOT_SET')}' → {USE_DEEP_FEATURES}")

N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

# Feature Bundle Version
BUNDLE_VERSION = "part1-v1"

# Ensure temp dir exists
os.makedirs(TEMP_DIR, exist_ok=True)
