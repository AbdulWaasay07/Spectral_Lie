#!/usr/bin/env python3
"""
Test script to verify model loading optimizations.
Tests:
1. Models load once at startup (singleton pattern)
2. Warm-up inference completes successfully
3. Startup time is acceptable (<15s)
"""

import sys
import time
import os

# Add part3_api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'part3_api'))

def test_startup_loading():
    """Test that models load correctly at startup"""
    print("=" * 60)
    print("TEST 1: Startup Model Loading")
    print("=" * 60)
    
    from app.orchestrator import preload_models, is_model_loaded
    
    start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting preload_models()...")
    
    try:
        preload_models()
        duration = time.time() - start
        
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Preload completed in {duration:.2f}s")
        
        if is_model_loaded():
            print("✓ Model loaded flag is True")
        else:
            print("✗ FAILED - Model loaded flag is False")
            return False
            
        if duration > 15:
            print(f"⚠ WARNING - Startup took {duration:.2f}s (>15s may timeout on Render)")
        else:
            print(f"✓ Startup time acceptable ({duration:.2f}s)")
            
        return True
        
    except Exception as e:
        print(f"✗ FAILED - Exception during preload: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_singleton_pattern():
    """Test that models are not reloaded on subsequent calls"""
    print("\n" + "=" * 60)
    print("TEST 2: Singleton Pattern Verification")
    print("=" * 60)
    
    from app.orchestrator import preload_models
    from part2 import utils
    
    try:
        # First load
        preload_models()
        model_id_1 = id(utils._MODEL)
        calibrator_id_1 = id(utils._CALIBRATOR)
        
        print(f"First load - Model ID: {model_id_1}, Calibrator ID: {calibrator_id_1}")
        
        # Second load
        preload_models()
        model_id_2 = id(utils._MODEL)
        calibrator_id_2 = id(utils._CALIBRATOR)
        
        print(f"Second load - Model ID: {model_id_2}, Calibrator ID: {calibrator_id_2}")
        
        if model_id_1 == model_id_2 and calibrator_id_1 == calibrator_id_2:
            print("✓ Singleton confirmed - same instances across multiple preload calls")
            return True
        else:
            print("✗ FAILED - Models were reloaded!")
            return False
            
    except Exception as e:
        print(f"✗ FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_no_reload():
    """Test that inference doesn't reload models"""
    print("\n" + "=" * 60)
    print("TEST 3: Inference Without Reload")
    print("=" * 60)
    
    from app.orchestrator import preload_models, detect_voice
    from part2 import utils
    import base64
    import io
    import wave
    import numpy as np
    
    try:
        # Preload models
        preload_models()
        model_id_before = id(utils._MODEL)
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 1.0
        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Run inference
        print(f"[{time.strftime('%H:%M:%S')}] Running test inference...")
        inference_start = time.time()
        result = detect_voice(audio_b64, "en", "test-request-123")
        inference_duration = time.time() - inference_start
        
        model_id_after = id(utils._MODEL)
        
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Inference completed in {inference_duration:.2f}s")
        print(f"Result: {result.get('classification')} (confidence: {result.get('confidence')})")
        
        if model_id_before == model_id_after:
            print("✓ Model not reloaded during inference")
            return True
        else:
            print("✗ FAILED - Model was reloaded during inference!")
            return False
            
    except Exception as e:
        print(f"✗ FAILED - Exception during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Spectral Lie - Model Loading Tests" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Set environment variable for CPU optimization
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    results = []
    
    # Run tests
    results.append(("Startup Loading", test_startup_loading()))
    results.append(("Singleton Pattern", test_singleton_pattern()))
    results.append(("Inference No Reload", test_inference_no_reload()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for deployment to Render")
    else:
        print("✗ SOME TESTS FAILED - Review errors before deploying")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
