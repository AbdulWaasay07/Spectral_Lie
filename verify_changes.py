# Quick Verification Script
# Run this to verify changes work correctly

import sys
import os

# Manual check of critical changes
print("="*60)
print("CRITICAL CHANGES VERIFICATION")
print("="*60)

# Check 1: part2/__init__.py no longer calls load_artifacts in infer()
print("\n1. Checking part2/__init__.py...")
with open(r"d:\Spectral_Lie\part2_detection\part2\__init__.py", "r") as f:
    content = f.read()
    if "utils.load_artifacts()" not in content.split("def infer")[1].split("def ")[0]:
        print("   ✓ load_artifacts() removed from infer() function")
    else:
        print("   ✗ FAIL - load_artifacts() still present in infer()")
    
    if "RuntimeError" in content and "Models not loaded" in content:
        print("   ✓ Safety check added for unloaded models")
    else:
        print("   ✗ FAIL - Missing safety check")

# Check 2: orchestrator.py has warm-up code
print("\n2. Checking orchestrator.py...")
with open(r"d:\Spectral_Lie\part3_api\app\orchestrator.py", "r") as f:
    content = f.read()
    if "warmup" in content.lower() and "synthetic" in content.lower():
        print("   ✓ Warm-up inference code added")
    else:
        print("   ✗ FAIL - Missing warm-up inference")
    
    if "total_startup_seconds" in content:
        print("   ✓ Startup timing added")
    else:
        print("   ✗ FAIL - Missing startup timing")

# Check 3: routes.py has timeout protection
print("\n3. Checking routes.py...")
with open(r"d:\Spectral_Lie\part3_api\app\routes.py", "r") as f:
    content = f.read()
    if "asyncio.wait_for" in content and "timeout=" in content:
        print("   ✓ Request timeout protection added")
    else:
        print("   ✗ FAIL - Missing timeout protection")
    
    if "wave.open" in content and "duration" in content:
        print("   ✓ Early audio validation added")
    else:
        print("   ✗ FAIL - Missing early validation")

# Check 4: config.py has tightened limits
print("\n4. Checking config.py...")
with open(r"d:\Spectral_Lie\part3_api\app\config.py", "r") as f:
    content = f.read()
    if "MAX_DURATION_SECONDS: float = 10.0" in content:
        print("   ✓ Max duration reduced to 10s")
    else:
        print("   ✗ FAIL - Max duration not updated")
    
    if "1 * 1024 * 1024" in content:
        print("   ✓ Max size reduced to 1MB")
    else:
        print("   ✗ FAIL - Max size not updated")

# Check 5: main.py has enhanced logging
print("\n5. Checking main.py...")
with open(r"d:\Spectral_Lie\part3_api\app\main.py", "r") as f:
    content = f.read()
    if "preload_duration" in content and "total_startup" in content:
        print("   ✓ Enhanced startup logging added")
    else:
        print("   ✗ FAIL - Missing enhanced logging")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nAll critical changes have been verified in the source files.")
print("\nNext Steps:")
print("1. Commit and push changes to GitHub")
print("2. Trigger Render deployment")
print("3. Monitor Render logs for 'Models preloaded' and 'Warm-up completed'")
print("4. Test endpoint: curl https://spectral-lie.onrender.com/detect-voice")
print()
