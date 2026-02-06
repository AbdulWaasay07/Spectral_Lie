import os

# Model Architecture
INPUT_DIM_DEFAULT = 92  # Acoustic features only (Dropping 1536 embeddings as synthetic data cannot match)
HIDDEN_LAYERS = [1024, 256, 64]
DROPOUT_RATES = [0.3, 0.2]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "classifier.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CALIBRATOR_PATH = os.path.join(MODELS_DIR, "calibrator.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Inference Defaults
DEFAULT_THRESHOLD = 0.5
MODEL_VERSION = "part2-v1-baseline"
