import base64
import binascii
import os
import tempfile
import soundfile as sf
from pydub import AudioSegment
from . import config, utils

class ValidationError(Exception):
    pass

def decode_and_validate(audio_base64: str) -> tuple[str, dict]:
    """
    Decodes base64 string, saves to temp file, converts to 16kHz mono WAV,
    and validates constraints.
    
    Returns:
        path_to_wav (str): Path to the converted wav file.
        metadata (dict): Metadata including hash, duration, etc.
    """
    try:
        raw_data = base64.b64decode(audio_base64)
    except binascii.Error as e:
        raise ValidationError(f"Invalid base64 string: {e}")

    # File size check
    if len(raw_data) > config.MAX_FILE_SIZE_BYTES:
        raise ValidationError(f"File too large: {len(raw_data)} bytes (max {config.MAX_FILE_SIZE_BYTES})")

    # Traceability hash
    original_hash = utils.compute_hash(raw_data)
    
    # Save to temp MP3
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", dir=config.TEMP_DIR, delete=False) as tmp_mp3:
            tmp_mp3.write(raw_data)
            tmp_mp3_path = tmp_mp3.name
    except IOError as e:
        raise ValidationError(f"Failed to write temp file: {e}")

    # Convert to WAV (16kHz, Mono)
    wav_path = tmp_mp3_path.replace(".mp3", ".wav")
    try:
        # Using pydub to verify it's valid audio and convert
        audio = AudioSegment.from_file(tmp_mp3_path)
        audio = audio.set_frame_rate(config.SAMPLE_RATE).set_channels(1)
        
        # Optimization: Slice to first 5 seconds to speed up extraction and prevent timeouts
        # Most acoustic features stabilize within 5s; processing 30s+ is too slow on free tier
        if len(audio) > 5000:
            audio = audio[:5000]
            
        audio.export(wav_path, format="wav")
    except Exception as e:
        # Cleanup
        if os.path.exists(tmp_mp3_path):
            try:
                os.remove(tmp_mp3_path)
            except OSError:
                pass
        raise ValidationError(f"Audio conversion failed (corrupted file?): {e}")
    finally:
        # We can remove the mp3 now
        if os.path.exists(tmp_mp3_path):
            try:
                os.remove(tmp_mp3_path)
            except OSError as e:
                utils.logger.warning(f"Could not delete temp file {tmp_mp3_path}: {e}")

    # Duration check using soundfile
    try:
        # sf.info is faster than loading whole file
        info = sf.info(wav_path)
        duration = info.duration
        
        if not (config.MIN_DURATION_SECONDS <= duration <= config.MAX_DURATION_SECONDS):
            os.remove(wav_path)
            raise ValidationError(f"Duration {duration:.2f}s out of bounds ({config.MIN_DURATION_SECONDS}-{config.MAX_DURATION_SECONDS}s)")
            
        metadata = {
            "duration": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "original_hash": original_hash,
            "raw_size": len(raw_data)
        }
        
        utils.logger.info(f"Processed audio: {original_hash[:8]}... | Duration: {duration:.2f}s")
        return wav_path, metadata

    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise ValidationError(f"Validation failed: {e}")
