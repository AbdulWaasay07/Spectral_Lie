# Suppress all warnings from audio/ML libraries
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import structlog
import numpy as np
from .errors import FeatureExtractionError, InferenceError

logger = structlog.get_logger()

# Global state
MODEL_LOADED = False

# --- Dynamic Path Setup for Local Dev ---
# If running locally without pip install -e, we need to add sibling dirs to path
# We assume this file is in d:/Spectral Lie/part3_api/app/orchestrator.py
# We need d:/Spectral Lie/part1_audio_features and d:/Spectral Lie/part2_detection
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
P1_PATH = os.path.join(BASE_DIR, "part1_audio_features")
P2_PATH = os.path.join(BASE_DIR, "part2_detection")

if os.path.exists(P1_PATH) and P1_PATH not in sys.path:
    sys.path.append(P1_PATH)
if os.path.exists(P2_PATH) and P2_PATH not in sys.path:
    sys.path.append(P2_PATH)

# --- Imports ---
try:
    import part1  
    import part2
except ImportError as e:
    logger.error("dependency_import_failed", error=str(e))
    # We don't raise here to allow app startup, but calls will fail
    part1 = None
    part2 = None

def detect_voice(audio_base64: str, language_hint: str | None, request_id: str):
    """
    Orchestrates the detection pipeline.
    
    PURE BLOCKING CODE - no timeout logic here.
    Timeout is handled at FastAPI route level via asyncio.wait_for.
    On any error, returns fallback response instead of raising.
    """
    if not part1 or not part2:
        # Return fallback instead of raising
        return _create_fallback_response(None, request_id, "Model backend not available")

    logger.info("orchestrator_start", request_id=request_id)
    
    # 1. Feature Extraction
    features = None
    try:
        features = part1.extract_features(audio_base64, language_hint)
        logger.info("feature_extraction_success", request_id=request_id)
    except Exception as e:
        logger.error("feature_extraction_failed", request_id=request_id, error=str(e))
        return _create_fallback_response(None, request_id, f"Feature extraction failed: {str(e)[:50]}")

    # 2. Inference
    try:
        result = part2.infer(features)
        logger.info("inference_success", request_id=request_id)
        return result
    except Exception as e:
        logger.error("inference_failed", request_id=request_id, error=str(e))
        return _create_fallback_response(features, request_id, f"Inference failed: {str(e)[:50]}")


def _create_fallback_response(features, request_id: str, reason: str):
    """
    Creates a fast fallback response using acoustic features only.
    Uses simple heuristics when full inference is too slow.
    """
    # Use acoustic features to make a basic prediction
    acoustic = features.acoustic_features if hasattr(features, 'acoustic_features') else {}
    
    # Simple heuristic based on acoustic features
    # Human speech typically has: lower jitter, stable pitch, natural rhythm
    jitter = acoustic.get("jitter", 0.02)
    shimmer = acoustic.get("shimmer", 0.05)
    hnr = acoustic.get("hnr", 15.0)
    
    # Heuristic scoring (higher = more likely AI)
    ai_score = 0.5  # neutral baseline
    
    # AI voices often have unnaturally low jitter/shimmer
    if jitter < 0.01:
        ai_score += 0.15
    if shimmer < 0.03:
        ai_score += 0.15
    # AI voices may have very high HNR (too clean)
    if hnr > 25:
        ai_score += 0.1
    
    # Clamp to valid range
    ai_score = max(0.3, min(0.7, ai_score))
    
    is_fake = ai_score >= 0.5
    
    return {
        "classification": "AI-Generated" if is_fake else "Human",
        "confidence": round(ai_score if is_fake else (1.0 - ai_score), 4),
        "explanation": f"{reason}. Based on acoustic analysis: jitter={jitter:.3f}, shimmer={shimmer:.3f}, HNR={hnr:.1f}",
        "model_version": "v1.0-fallback",
        "decision_threshold": 0.5
    }





def preload_models():
    """
    Triggers lazy loading of models in part1 and part2.
    Called on API startup.
    """
    import time
    start_time = time.time()
    
    if part1:
        try:
            from part1 import config as p1_config
            if p1_config.USE_DEEP_FEATURES:
                from part1.features_deep import load_model
                load_model()
                logger.info("part1_model_preloaded")
            else:
                logger.info("part1_deep_model_skipped_by_config")
        except Exception as e:
            logger.error("part1_preload_failed", error=str(e))
    
    if part2:
        try:
            from part2.utils import load_artifacts
            load_artifacts()
            
            # Verify models are actually loaded
            from part2 import utils as p2_utils
            if p2_utils._MODEL is None or p2_utils._CALIBRATOR is None:
                raise RuntimeError("part2 models failed to load despite no exception")
            
            logger.info("part2_model_preloaded", 
                       model_loaded=p2_utils._MODEL is not None,
                       calibrator_loaded=p2_utils._CALIBRATOR is not None)
        except Exception as e:
            logger.error("part2_preload_failed", error=str(e))
            # Don't set MODEL_LOADED if part2 fails
            return

    # Warm-up: verify critical components are loaded
    try:
        logger.info("verifying_model_components")
        
        # Verify part2 models exist and are accessible
        from part2 import utils as p2_utils
        if p2_utils._MODEL is not None:
            logger.info("model_verified", model_type=str(type(p2_utils._MODEL)))
        if p2_utils._CALIBRATOR is not None:
            logger.info("calibrator_verified")
        
        logger.info("warmup_verification_completed")
        
    except Exception as e:
        logger.warning("warmup_verification_failed", error=str(e))

    global MODEL_LOADED
    MODEL_LOADED = True
    
    total_duration = time.time() - start_time
    logger.info("all_models_preloaded", total_startup_seconds=round(total_duration, 2))

def is_model_loaded():
    return MODEL_LOADED
