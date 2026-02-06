# Suppress torch/numpy warnings
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Union
import torch
import numpy as np

# Try implicit relative import if part1 installed, else define Protocol
try:
    from part1.bundle import FeatureBundle
except ImportError:
    # Fallback definition for type hinting if part1 not installed in env
    from dataclasses import dataclass
    @dataclass
    class FeatureBundle:
        acoustic_features: Dict[str, float]
        deep_embeddings: np.ndarray
        metadata: Dict[str, Any]
        version: str

from . import utils, explain, config

def infer(features: FeatureBundle) -> Dict[str, Any]:
    """
    Input: FeatureBundle (part1 output)
    Output: DetectionResult JSON
    """
    # 1. Verify models are loaded (should be loaded at startup via orchestrator.preload_models())
    if utils._MODEL is None or utils._CALIBRATOR is None:
        raise RuntimeError(
            "Models not loaded. Ensure orchestrator.preload_models() was called at startup."
        )
    
    # 2. Preprocess
    # Note: real robustness requires checking input dimensions against model expectation
    input_tensor = utils.prepare_input(features)
    
    # 3. Predict & Calibrate
    with torch.no_grad():
        logits = utils._MODEL(input_tensor)
        proba = utils._CALIBRATOR.predict_proba(logits).item()
        
    # 4. Explain
    # Threshold check
    is_fake = proba >= config.DEFAULT_THRESHOLD
    explanation_text = explain.generate_explanation(
        features.acoustic_features, 
        utils._BASELINES, 
        proba, 
        config.DEFAULT_THRESHOLD
    )
    
    # 5. Result
    winner_proba = proba if is_fake else (1.0 - proba)
    
    return {
        "classification": "AI-Generated" if is_fake else "Human",
        "confidence": round(float(winner_proba), 4),
        "explanation": explanation_text,
        "model_version": config.MODEL_VERSION,
        "decision_threshold": config.DEFAULT_THRESHOLD
    }
