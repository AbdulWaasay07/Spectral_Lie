import numpy as np
import librosa
from . import config, utils

def preprocess_audio(wav_path: str) -> np.ndarray:
    """
    Loads WAV, ensures 16kHz mono, trims silence, and normalizes loudness.
    Returns float32 numpy array.
    """
    try:
        # Load audio (already converted to 16k mono by io.py, but safe reload)
        y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        
        # 1. Trim Silence
        # top_db=60 is standard, but we want to be conservative to keep micro-pauses
        # Prompt says "low amplitude threshold but conservative"
        y_trimmed, _ = librosa.effects.trim(y, top_db=50) 
        
        # 2. Loudness Normalization (RMS)
        # Target RMS: -23 LUFS approx or simple RMS fixed value.
        # We will use simple RMS normalization to a target level.
        target_rms = 0.05  # Experimentally decent value
        current_rms = np.sqrt(np.mean(y_trimmed**2))
        
        if current_rms > 0:
            scale_factor = target_rms / current_rms
            y_norm = y_trimmed * scale_factor
        else:
            y_norm = y_trimmed

        # 3. Clipping Guard
        # Clip to -1.0 to 1.0 range
        y_final = np.clip(y_norm, -1.0, 1.0)
        
        # Ensure float32
        y_final = y_final.astype(np.float32)
        
        return y_final

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed for {wav_path}: {e}")
