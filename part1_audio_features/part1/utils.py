import logging
import hashlib
import sys

def setup_logger(name: str = "part1_audio_features") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def compute_hash(data: bytes) -> str:
    """Computes SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()

logger = setup_logger()
