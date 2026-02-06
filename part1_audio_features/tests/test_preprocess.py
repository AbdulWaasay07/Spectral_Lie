import numpy as np
from part1 import preprocess

def test_preprocess_output_shape(sample_wav_path):
    waveform = preprocess.preprocess_audio(sample_wav_path)
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    # Check normalization range
    assert np.max(np.abs(waveform)) <= 1.0
