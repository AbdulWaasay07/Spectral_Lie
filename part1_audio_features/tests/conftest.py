import pytest
import numpy as np
import base64
import os
import tempfile
import soundfile as sf
from pydub import AudioSegment

@pytest.fixture(scope="session")
def sample_wav_path():
    """Creates a temporary 16kHz sine wave WAV file."""
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz sine
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        path = f.name
        
    yield path
    
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture(scope="session")
def sample_mp3_base64(sample_wav_path):
    """Converts the sample WAV to MP3 and returns base64 string."""
    wav_audio = AudioSegment.from_wav(sample_wav_path)
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        wav_audio.export(f.name, format="mp3")
        mp3_path = f.name
        
    with open(mp3_path, "rb") as f:
        data = f.read()
        
    os.remove(mp3_path)
    return base64.b64encode(data).decode("utf-8")
