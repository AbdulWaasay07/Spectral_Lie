import os
import tempfile
import base64
import uuid
import structlog
from .errors import ValidationError

logger = structlog.get_logger()

def decode_audio(base64_string: str) -> str:
    """
    Decodes base64 string to a temporary file.
    Returns the path to the temporary file.
    CALLER IS RESPONSIBLE FOR DELETING THE FILE.
    """
    try:
        # Basic validation
        if "," in base64_string:
            # Remove header if present (e.g., "data:audio/mp3;base64,...")
            base64_string = base64_string.split(",")[1]
            
        audio_data = base64.b64decode(base64_string)
        
        # Create temp file
        # We use delete=False because we need to pass the path to Part 1
        # Part 1 uses librosa/soundfile which needs a path
        fd, path = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(audio_data)
            
        return path
    except Exception as e:
        logger.error("audio_decode_failed", error=str(e))
        raise ValidationError("Invalid Base64 audio string")

def cleanup_file(path: str):
    """Safely deletes a file."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning("cleanup_failed", path=path, error=str(e))
