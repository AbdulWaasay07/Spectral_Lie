import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app import auth, rate_limiter, orchestrator

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_redis():
    mock = AsyncMock()
    mock.incr.return_value = 1
    mock.expire.return_value = True
    mock.ping.return_value = True
    
    # Patch the global redis_client in rate_limiter
    rate_limiter.redis_client = mock
    return mock

@pytest.fixture
def mock_backend(monkeypatch):
    """Mocks Part 1 and Part 2 backend calls."""
    mock_p1 = MagicMock()
    # Mock extract_features return value
    # It returns a FeatureBundle-like object
    mock_bundle = MagicMock()
    mock_bundle.acoustic_features = {"jitter": 0.001}
    mock_p1.extract_features.return_value = mock_bundle
    
    mock_p2 = MagicMock()
    mock_p2.infer.return_value = {
        "classification": "AI-generated",
        "confidence": 0.99,
        "explanation": "High stability detected.",
        "model_version": "v1.0"
    }

    # Patch the modules in orchestrator
    monkeypatch.setattr(orchestrator, "part1", mock_p1)
    monkeypatch.setattr(orchestrator, "part2", mock_p2)
    
    return mock_p1, mock_p2
