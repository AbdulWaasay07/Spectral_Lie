# Suppress all warnings from audio/ML libraries
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import structlog
import numpy as np
from .errors import FeatureExtractionError, InferenceError
from . import fast_gate

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
    Orchestrates the detection pipeline with ULTRA-FAST PRE-GATE.
    
    Order of operations:
    1. FAST GATE (NumPy only, <100ms) - runs BEFORE any heavy computation
    2. Feature Extraction (only if gate inconclusive)
    3. ML Model Inference (only if features succeeded)
    
    On ANY error → Human-biased fallback (never AI).
    """
    logger.info("orchestrator_start", request_id=request_id)
    
    # ============================================
    # STEP 1: ULTRA-FAST PRE-GATE (NumPy only)
    # Runs BEFORE any librosa/torch - target <100ms
    # ============================================
    fast_gate_hint = None  # Track fast gate result for fallback
    try:
        gate_result = fast_gate.check(audio_base64)
        if gate_result is not None:
            confidence = gate_result.get("confidence", 0.85)
            features = gate_result.get("features", {})
            is_human = gate_result.get("is_human", True)
            
            if is_human:
                logger.info("ultra_fast_gate_human", request_id=request_id, confidence=confidence)
                return {
                    "classification": "Human",
                    "confidence": round(confidence, 3),
                    "explanation": f"Fast acoustic gate: human speech detected (ZCR={features.get('zcr', 0):.3f}, silence={features.get('silence_ratio', 0):.2f}).",
                    "model_version": "v1.2-fast-gate",
                    "decision_threshold": 0.5
                }
            else:
                # Fast gate detected AI
                logger.info("ultra_fast_gate_ai", request_id=request_id, confidence=confidence)
                return {
                    "classification": "AI-Generated",
                    "confidence": round(confidence, 3),
                    "explanation": f"Fast acoustic gate: AI speech detected (ZCR={features.get('zcr', 0):.3f}, RMS_var={features.get('rms_variance', 0):.5f}).",
                    "model_version": "v1.2-fast-gate",
                    "decision_threshold": 0.5
                }
        logger.info("fast_gate_inconclusive", request_id=request_id)
    except Exception as e:
        logger.warning("fast_gate_error", request_id=request_id, error=str(e))
        # Continue to next stage
    
    # ============================================
    # STEP 2: FEATURE EXTRACTION (only if gate inconclusive)
    # ============================================
    if not part1 or not part2:
        return _create_fallback_response(None, request_id, "Model backend not available", fast_gate_hint)
    
    features = None
    try:
        features = part1.extract_features(audio_base64, language_hint)
        logger.info("feature_extraction_success", request_id=request_id)
    except Exception as e:
        logger.error("feature_extraction_failed", request_id=request_id, error=str(e))
        return _create_fallback_response(None, request_id, f"Feature extraction failed: {str(e)[:50]}", fast_gate_hint)

    # ---- POST-EXTRACTION GATE (BALANCED) ----
    # Classify based on acoustic features - can detect BOTH Human and AI
    acoustic = features.acoustic_features if hasattr(features, 'acoustic_features') else {}
    jitter = acoustic.get("jitter_local", acoustic.get("jitter", 0))
    shimmer = acoustic.get("shimmer_local", acoustic.get("shimmer", 0))
    hnr = acoustic.get("hnr", 20)  # Default to neutral HNR
    
    # AI scoring: very low jitter, very low shimmer, very high HNR
    ai_score = 0
    if jitter < 0.005:  # AI has very stable pitch
        ai_score += 1
    if shimmer < 0.02:  # AI has very stable amplitude
        ai_score += 1
    if hnr > 28:  # AI is very clean (high HNR)
        ai_score += 1
    
    # Human scoring: natural variation in jitter/shimmer
    human_score = 0
    if jitter > 0.01:  # Humans have natural jitter
        human_score += 1
    if shimmer > 0.03:  # Humans have natural shimmer
        human_score += 1
    if hnr < 25:  # Humans have moderate HNR
        human_score += 1
    
    # If strong AI indicators → classify as AI
    if ai_score >= 2 and human_score == 0:
        logger.info("post_extraction_gate_ai", request_id=request_id, 
                   jitter=jitter, shimmer=shimmer, hnr=hnr, ai_score=ai_score)
        return {
            "classification": "AI-Generated",
            "confidence": min(0.90, 0.70 + ai_score * 0.1),
            "explanation": f"AI acoustic signature (jitter={jitter:.3f}, shimmer={shimmer:.3f}, HNR={hnr:.1f}).",
            "model_version": "v1.2-acoustic-gate",
            "decision_threshold": 0.5
        }
    
    # If strong Human indicators → classify as Human
    if human_score >= 2:
        logger.info("post_extraction_gate_human", request_id=request_id, 
                   jitter=jitter, shimmer=shimmer, hnr=hnr, human_score=human_score)
        return {
            "classification": "Human",
            "confidence": min(0.90, 0.70 + human_score * 0.1),
            "explanation": f"Human acoustic signature (jitter={jitter:.3f}, shimmer={shimmer:.3f}, HNR={hnr:.1f}).",
            "model_version": "v1.2-acoustic-gate",
            "decision_threshold": 0.5
        }

    # ============================================
    # STEP 3: ML MODEL INFERENCE (last resort)
    # ============================================
    try:
        result = part2.infer(features)
        logger.info("inference_success", request_id=request_id)
        return result
    except Exception as e:
        logger.error("inference_failed", request_id=request_id, error=str(e))
        return _create_fallback_response(features, request_id, f"Inference failed: {str(e)[:50]}", fast_gate_hint)


def _create_fallback_response(features, request_id: str, reason: str, fast_gate_hint=None):
    """
    Creates a fallback response using acoustic features.
    Respects fast_gate_hint if available, otherwise uses acoustic analysis.
    """
    acoustic = features.acoustic_features if features and hasattr(features, 'acoustic_features') else {}
    
    # Get acoustic features with neutral defaults
    jitter = acoustic.get("jitter_local", acoustic.get("jitter", 0.02))
    shimmer = acoustic.get("shimmer_local", acoustic.get("shimmer", 0.04))
    hnr = acoustic.get("hnr", 20.0)
    
    # If fast gate had a hint, respect it for the fallback
    if fast_gate_hint is not None:
        if fast_gate_hint == "AI":
            logger.info("fallback_using_ai_hint", request_id=request_id)
            return {
                "classification": "AI-Generated",
                "confidence": 0.65,
                "explanation": f"{reason}. Earlier acoustic analysis suggested AI-generated audio.",
                "model_version": "v1.2-fallback-ai",
                "decision_threshold": 0.5
            }
    
    # AI scoring: very clean/stable features
    ai_score = 0
    if jitter < 0.008:
        ai_score += 1
    if shimmer < 0.025:
        ai_score += 1
    if hnr > 26:
        ai_score += 1
    
    # Human scoring: natural variation
    human_score = 0
    if jitter > 0.015:
        human_score += 1
    if shimmer > 0.04:
        human_score += 1
    if hnr < 24:
        human_score += 1
    
    # Decide based on scores
    if ai_score >= 2 and human_score == 0:
        return {
            "classification": "AI-Generated",
            "confidence": round(0.55 + ai_score * 0.1, 3),
            "explanation": f"{reason}. Acoustic indicators suggest AI (jitter={jitter:.3f}, shimmer={shimmer:.3f}, HNR={hnr:.1f}).",
            "model_version": "v1.2-fallback-ai",
            "decision_threshold": 0.5
        }
    else:
        # Default to Human when uncertain
        confidence = 0.55 + human_score * 0.1
        confidence = min(0.85, max(0.55, confidence))
        return {
            "classification": "Human",
            "confidence": round(confidence, 3),
            "explanation": f"{reason}. Acoustic indicators suggest human (jitter={jitter:.3f}, shimmer={shimmer:.3f}, HNR={hnr:.1f}).",
            "model_version": "v1.2-fallback-human",
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
