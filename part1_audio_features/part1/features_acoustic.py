import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from . import config, utils

def extract_acoustic_features(waveform: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    """
    Extracts interpretable acoustic features: MFCC, Pitch, Jitter, Shimmer, HNR, Spectral stats.
    Returns: dictionary of float values.
    """
    features = {}
    
    # --- Librosa Features ---
    
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    for i in range(config.N_MFCC):
        features[f"mfcc_mean_{i}"] = float(np.mean(mfcc[i]))
        features[f"mfcc_std_{i}"] = float(np.std(mfcc[i]))
        
    # Delta and Delta-Delta omitted for brevity/speed unless strictly required, 
    # but the prompt asked for them: "with delta and delta-delta"
    # Let's add them.
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(config.N_MFCC):
        features[f"mfcc_delta_mean_{i}"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_std_{i}"] = float(np.std(mfcc_delta[i]))
        features[f"mfcc_delta2_mean_{i}"] = float(np.mean(mfcc_delta2[i]))
        features[f"mfcc_delta2_std_{i}"] = float(np.std(mfcc_delta2[i]))

    # 2. Spectral Features
    cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(cent))
    features["spectral_centroid_std"] = float(np.std(cent))
    
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))
    
    flatness = librosa.feature.spectral_flatness(y=waveform)
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))
    
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # --- Parselmouth (Praat) Features ---
    try:
        sound = parselmouth.Sound(waveform, sampling_frequency=sr)
        
        # Pitch (F0) with optimized range for speed
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        pitch_values = pitch.selected_array['frequency']
        # Filter 0 (unvoiced) and outliers
        pitch_values_voiced = pitch_values[(pitch_values > 75) & (pitch_values < 500)]
        
        if len(pitch_values_voiced) > 0:
            features["pitch_mean"] = float(np.mean(pitch_values_voiced))
            features["pitch_std"] = float(np.std(pitch_values_voiced))
            features["voiced_ratio"] = float(len(pitch_values_voiced) / len(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["voiced_ratio"] = 0.0

        # Jitter (Optimized for speed)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter_local"] = call(pointProcess, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        
        # Shimmer (Optimized for speed)
        features["shimmer_local"] = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr"] = call(harmonicity, "Get mean", 0, 0)
        
    except Exception as e:
        utils.logger.warning(f"Praat feature extraction failed: {e}")
        features["jitter_local"] = None
        features["shimmer_local"] = None
        features["hnr"] = None
        
    return features
