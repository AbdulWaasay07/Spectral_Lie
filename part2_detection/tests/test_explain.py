import pytest
from part2 import explain

def test_low_jitter_explanation():
    # Mock data
    features = {"jitter_local": 0.001} # Very low
    baselines = {"jitter_local": {"median": 0.02}} # Human median
    
    # Fake call
    text = explain.generate_explanation(features, baselines, confidence=0.9, threshold=0.5)
    
    assert "Unusually low jitter" in text
    assert "AI-generated" in text

def test_human_verdict():
    features = {"jitter_local": 0.02}
    baselines = {"jitter_local": {"median": 0.02}}
    
    text = explain.generate_explanation(features, baselines, confidence=0.2, threshold=0.5)
    
    assert "Human" in text
    assert "typical human ranges" in text
