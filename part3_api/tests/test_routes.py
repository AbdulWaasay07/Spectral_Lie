from app.config import settings

def test_health_check(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_detect_voice_unauthorized(client):
    # Missing header
    response = client.post("/detect-voice", json={"audio_base64": "invalid"})
    assert response.status_code == 401 # Should be 403 or 401 depending on auto_error

    # Invalid key
    response = client.post(
        "/detect-voice", 
        headers={settings.API_KEY_HEADER: "wrong-key"},
        json={"audio_base64": "invalid"}
    )
    assert response.status_code == 401

def test_detect_voice_success(client, mock_redis, mock_backend):
    # Valid request
    api_keys = settings.API_KEYS.split(",")
    valid_key = api_keys[0] if api_keys else "test-key-123"
    
    payload = {
        "audio_base64": "SUQzBAAAAAAAI1...",  # Dummy base64
        "language": "en"
    }
    
    response = client.post(
        "/detect-voice",
        headers={settings.API_KEY_HEADER: valid_key},
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "AI-generated"
    assert data["confidence"] == 0.99
    assert "request_id" in data
    
    # Verify backend called
    mock_p1, mock_p2 = mock_backend
    mock_p1.extract_features.assert_called_once()
    mock_p2.infer.assert_called_once()

def test_detect_voice_rate_limit(client, mock_redis, mock_backend):
    # Simulate rate limit exceeded
    mock_redis.incr.return_value = settings.RATE_LIMIT_PER_MINUTE + 1
    
    api_keys = settings.API_KEYS.split(",")
    valid_key = api_keys[0]
    
    response = client.post(
        "/detect-voice",
        headers={settings.API_KEY_HEADER: valid_key},
        json={"audio_base64": "dummy"}
    )
    
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
