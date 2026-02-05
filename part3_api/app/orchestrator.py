import sys
import os
import structlog
from .errors import FeatureExtractionError, InferenceError

logger = structlog.get_logger()

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
    """
    if not part1 or not part2:
        raise InferenceError("Model backend not available.")

    logger.info("orchestrator_start", request_id=request_id)

    # 1. Feature Extraction (Part 1)
    try:
        # Part 1 extract_features accepts base64 directly
        features = part1.extract_features(audio_base64, language_hint)
        logger.info("feature_extraction_success", request_id=request_id)
    except Exception as e:
        logger.error("feature_extraction_failed", request_id=request_id, error=str(e))
        raise FeatureExtractionError(str(e))

    # 2. Inference (Part 2)
    try:
        result = part2.infer(features)
        
        # Inject request_id into result if not present
        result["request_id"] = request_id
        
        logger.info("inference_success", request_id=request_id, classification=result.get("classification"))
        return result
    except Exception as e:
        logger.error("inference_failed", request_id=request_id, error=str(e))
        raise InferenceError(str(e))

def preload_models():
    """
    Triggers lazy loading of models in part1 and part2.
    Called on API startup.
    """
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
            logger.info("part2_model_preloaded")
        except Exception as e:
            logger.error("part2_preload_failed", error=str(e))
