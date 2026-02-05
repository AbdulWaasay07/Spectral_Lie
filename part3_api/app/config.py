import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Spectral Lie Detector API"
    DEBUG: bool = False
    
    # Secrets
    API_KEY_HEADER: str = "x-api-key"
    # In production, this should be a list or loaded from a secure store
    # For buildathon, we accept a single key or a comma-separated list
    API_KEYS: str = "test-key-123" 
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Validation
    MAX_AUDIO_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
    MIN_DURATION_SECONDS: float = 3.0
    MAX_DURATION_SECONDS: float = 30.0
    
    # Model Paths (optional, can fallback to hardcoded defaults in Part 1/2)
    # These env vars allow us to override paths if needed in Docker
    PART1_ARTIFACTS_DIR: str | None = None
    PART2_MODEL_PATH: str | None = None
    
    class Config:
        env_file = ".env"

settings = Settings()
