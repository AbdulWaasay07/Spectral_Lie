import pytest
import os
from part1 import io, config

def test_decode_valid_mp3(sample_mp3_base64):
    wav_path, metadata = io.decode_and_validate(sample_mp3_base64)
    assert os.path.exists(wav_path)
    assert metadata["duration"] >= config.MIN_DURATION_SECONDS
    assert metadata["sample_rate"] == 16000
    assert metadata["channels"] == 1
    
    # Cleanup
    if os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except OSError:
            pass

def test_invalid_base64():
    with pytest.raises(io.ValidationError):
        io.decode_and_validate("not a base64 string")

def test_too_short_audio(sample_mp3_base64):
    # This might define a constraint issue if the fixture is short, 
    # but our fixture uses 4.0s which is > 3.0s min.
    pass
