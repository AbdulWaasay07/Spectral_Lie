import numpy as np
import pytest
from part1 import features_acoustic, features_deep

@pytest.fixture
def mock_waveform():
    # 4 seconds of random noise at 16kHz
    return np.random.uniform(-0.5, 0.5, 16000 * 4).astype(np.float32)

def test_acoustic_features(mock_waveform):
    features = features_acoustic.extract_acoustic_features(mock_waveform)
    
    expected_keys = [
        "mfcc_mean_0", "pitch_mean", "jitter_local", "shimmer_local", "hnr",
        "spectral_centroid_mean", "zcr_mean"
    ]
    
    for k in expected_keys:
        assert k in features, f"Missing key: {k}"
        # Some might be None if extraction failed (e.g. pitch on noise), but typically for noise pitch is 0.
        # Jitter/Shimmer might be None if no pitch.
        
def test_deep_embeddings(mock_waveform):
    # This test might be slow as it loads the model
    # We can mock the model loading if needed, but integration test is better.
    try:
        emb = features_deep.extract_deep_embeddings(mock_waveform)
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32
        # Dim = 768 * 2 = 1536
        assert emb.shape == (1536,)
    except Exception as e:
        pytest.fail(f"Deep feature extraction failed: {e}")
