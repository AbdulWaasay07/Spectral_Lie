import torch
import json
import os
import joblib
import numpy as np
from types import SimpleNamespace
from . import config, model, calibrator

# Caches
_MODEL = None
_SCALER = None
_CALIBRATOR = None
_BASELINES = None

def load_artifacts():
    """Lazily loads model, scaler, calibrator, and baselines."""
    global _MODEL, _SCALER, _CALIBRATOR, _BASELINES
    
    # 1. Load Model
    if _MODEL is None:
        # Initialize architecture
        _MODEL = model.SimpleClassifier(config.INPUT_DIM_DEFAULT)
        # Load weights if exist
        if os.path.exists(config.DEFAULT_MODEL_PATH):
            _MODEL.load_state_dict(torch.load(config.DEFAULT_MODEL_PATH, map_location="cpu"))
        _MODEL.eval()

    # 2. Load Scaler
    if _SCALER is None and os.path.exists(config.SCALER_PATH):
        _SCALER = joblib.load(config.SCALER_PATH)

    # 3. Load Calibrator
    if _CALIBRATOR is None:
        _CALIBRATOR = calibrator.TemperatureScaler()
        if os.path.exists(config.CALIBRATOR_PATH):
            state = torch.load(config.CALIBRATOR_PATH, map_location="cpu")
            _CALIBRATOR.load_state_dict(state)
        _CALIBRATOR.eval()

    # 4. Load Human Baselines
    # Assuming Part 1's baseline path; ideally this should be copied to models/
    # But for now we try to find it relative to part1
    if _BASELINES is None:
        baseline_path = os.path.abspath(os.path.join(config.BASE_DIR, "../../part1_audio_features/baselines/human_baseline.json"))
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                _BASELINES = json.load(f)
        else:
            _BASELINES = {} # Fallback

def prepare_input(feature_bundle) -> torch.Tensor:
    """Concatenates embeddings and acoustic features into a tensor."""
    # This logic needs to align with config.INPUT_DIM_DEFAULT
    # 1. Deep Embeddings (Ignored for now due to synthetic mismatch)
    # emb = feature_bundle.deep_embeddings
    
    # 2. Acoustic Features (Order matters!)
    ac_dict = feature_bundle.acoustic_features
    ac_keys = sorted(ac_dict.keys())
    ac_vals = np.array([ac_dict[k] for k in ac_keys], dtype=np.float32)
    
    # Concatenate (Acoustic only)
    combined = ac_vals
    
    # Normalize if scaler exists
    if _SCALER:
        combined = _SCALER.transform(combined.reshape(1, -1)).flatten()
        
    return torch.from_numpy(combined).float().unsqueeze(0) # (1, D)
