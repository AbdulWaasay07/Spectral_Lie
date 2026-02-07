"""
Ultra-Fast Pre-Gate for Audio Classification

This module runs BEFORE any heavy computation (librosa, torch, etc.).
Uses only base64 decode + NumPy for O(n) performance.
Target: < 100ms execution time.
"""

import base64
import io
import numpy as np
import struct
import structlog

log = structlog.get_logger()

# Thresholds for human speech detection
DURATION_MAX = 5.0  # seconds - reject longer audio
SILENCE_RATIO_MAX = 0.6  # max silence allowed
ZCR_MIN = 0.02  # minimum zero-crossing rate for speech
RMS_VARIANCE_MIN = 0.001  # minimum energy variance for speech


def decode_audio_fast(audio_base64: str) -> tuple[np.ndarray, int] | None:
    """
    Fast audio decode using only standard library.
    Returns (samples, sample_rate) or None on failure.
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        # Try to detect WAV format
        if audio_bytes[:4] == b'RIFF':
            return _decode_wav(audio_bytes)
        
        # For other formats, return None (will fall through to heavy processing)
        return None
        
    except Exception as e:
        log.warning("fast_gate_decode_failed", error=str(e))
        return None


def _decode_wav(audio_bytes: bytes) -> tuple[np.ndarray, int] | None:
    """Decode WAV file using struct (no external dependencies)."""
    try:
        # Parse WAV header
        if len(audio_bytes) < 44:
            return None
            
        # Get sample rate (bytes 24-27)
        sample_rate = struct.unpack('<I', audio_bytes[24:28])[0]
        
        # Get bits per sample (bytes 34-35)
        bits_per_sample = struct.unpack('<H', audio_bytes[34:36])[0]
        
        # Get number of channels (bytes 22-23)
        num_channels = struct.unpack('<H', audio_bytes[22:24])[0]
        
        # Find data chunk
        data_start = 44  # Standard WAV header size
        data_bytes = audio_bytes[data_start:]
        
        # Convert to numpy array
        if bits_per_sample == 16:
            samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif bits_per_sample == 8:
            samples = (np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            return None
        
        # Convert to mono if stereo
        if num_channels == 2:
            samples = samples[::2]  # Take left channel only
            
        return samples, sample_rate
        
    except Exception:
        return None


def compute_features_fast(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Compute acoustic features using only NumPy.
    All operations are O(n) and vectorized.
    """
    duration = len(samples) / sample_rate
    
    # RMS energy
    rms = np.sqrt(np.mean(samples ** 2))
    
    # RMS variance (chunked)
    chunk_size = sample_rate // 10  # 100ms chunks
    if len(samples) > chunk_size:
        n_chunks = len(samples) // chunk_size
        chunks = samples[:n_chunks * chunk_size].reshape(-1, chunk_size)
        chunk_rms = np.sqrt(np.mean(chunks ** 2, axis=1))
        rms_variance = np.var(chunk_rms)
    else:
        rms_variance = 0.0
    
    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / 2
    zcr = zero_crossings / len(samples)
    
    # Silence ratio (samples below threshold)
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(samples) < silence_threshold) / len(samples)
    
    return {
        "duration": duration,
        "rms": float(rms),
        "rms_variance": float(rms_variance),
        "zcr": float(zcr),
        "silence_ratio": float(silence_ratio)
    }


def check(audio_base64: str) -> dict | None:
    """
    Ultra-fast pre-gate check.
    
    Returns:
        - {"is_human": True, "confidence": float, "features": dict} if confident human
        - {"is_human": False, "confidence": float, "features": dict} if confident AI
        - None if inconclusive (needs heavy processing)
    """
    # Step 1: Fast decode
    result = decode_audio_fast(audio_base64)
    if result is None:
        # Can't decode fast - fall through to heavy processing
        log.info("fast_gate_decode_skip", reason="unsupported_format")
        return None
    
    samples, sample_rate = result
    
    # Step 2: Compute features
    features = compute_features_fast(samples, sample_rate)
    
    log.info("fast_gate_features", **features)
    
    # Step 3: Check duration limit
    if features["duration"] > DURATION_MAX:
        log.info("fast_gate_skip", reason="duration_exceeded", duration=features["duration"])
        return None  # Too long, needs proper analysis
    
    # Step 4: Human speech detection
    # Human speech has: energy variance, moderate ZCR, low silence ratio
    is_likely_human = (
        features["rms_variance"] > RMS_VARIANCE_MIN and
        features["zcr"] > ZCR_MIN and
        features["silence_ratio"] < SILENCE_RATIO_MAX
    )
    
    if is_likely_human:
        # Calculate confidence based on how strongly features indicate human
        confidence = 0.75
        if features["zcr"] > 0.05:
            confidence += 0.05
        if features["rms_variance"] > 0.005:
            confidence += 0.05
        if features["silence_ratio"] < 0.3:
            confidence += 0.05
        
        confidence = min(0.90, confidence)
        
        log.info("fast_gate_human_detected", confidence=confidence)
        return {
            "is_human": True,
            "confidence": confidence,
            "features": features
        }
    
    # Step 5: AI speech detection
    # AI-generated speech typically has:
    # - Very low RMS variance (too stable/uniform energy)
    # - Very low ZCR (over-smoothed waveform)
    # - Very low silence ratio (continuous, no natural pauses)
    AI_RMS_VARIANCE_MAX = 0.0005  # AI is too stable
    AI_ZCR_MAX = 0.015           # AI is over-smoothed
    AI_SILENCE_RATIO_MAX = 0.15  # AI has fewer natural pauses
    
    is_likely_ai = (
        features["rms_variance"] < AI_RMS_VARIANCE_MAX and
        features["zcr"] < AI_ZCR_MAX and
        features["silence_ratio"] < AI_SILENCE_RATIO_MAX
    )
    
    if is_likely_ai:
        # Calculate confidence based on how strongly features indicate AI
        confidence = 0.75
        if features["rms_variance"] < 0.0002:
            confidence += 0.05
        if features["zcr"] < 0.01:
            confidence += 0.05
        if features["silence_ratio"] < 0.1:
            confidence += 0.05
        
        confidence = min(0.90, confidence)
        
        log.info("fast_gate_ai_detected", confidence=confidence)
        return {
            "is_human": False,
            "confidence": confidence,
            "features": features
        }
    
    # Not confident - fall through to heavy processing
    log.info("fast_gate_inconclusive")
    return None
