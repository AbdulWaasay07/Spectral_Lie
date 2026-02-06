import sys
import numpy as np
import json
from part2 import infer
# Mock the class if Part 1 not installed, else use it
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FeatureBundle:
    acoustic_features: Dict[str, float]
    deep_embeddings: np.ndarray
    metadata: Dict[str, Any]
    version: str = "part1-v1"

def main():
    print("Creating dummy FeatureBundle input...")
    
    # 1. Random acoustic features matching Part 1's exact set (92 features)
    acoustic = {}
    
    # MFCCs (0-12) x 3 orders (raw, delta, delta2) x 2 stats (mean, std) = 78 features
    for i in range(13):
        for stat in ["mean", "std"]:
            acoustic[f"mfcc_{stat}_{i}"] = np.random.rand()
            acoustic[f"mfcc_delta_{stat}_{i}"] = np.random.rand()
            acoustic[f"mfcc_delta2_{stat}_{i}"] = np.random.rand()
            
    # Spectral (4 types * 2 stats) = 8 features
    for feat in ["spectral_centroid", "spectral_rolloff", "spectral_flatness", "zcr"]:
        acoustic[f"{feat}_mean"] = np.random.rand() * 1000
        acoustic[f"{feat}_std"] = np.random.rand() * 100

    # Prosodic (6 features)
    acoustic["pitch_mean"] = 120.0
    acoustic["pitch_std"] = 5.0
    acoustic["voiced_ratio"] = 0.8
    acoustic["jitter_local"] = 0.0005
    acoustic["shimmer_local"] = 0.01
    acoustic["hnr"] = 35.0
    
    # Total keys should be 78 + 8 + 6 = 92

    
    # 2. Random deep embeddings (Part 1 size: 1536)
    embeddings = np.random.randn(1536).astype(np.float32)
    
    bundle = FeatureBundle(
        acoustic_features=acoustic,
        deep_embeddings=embeddings,
        metadata={"duration": 4.0},
        version="v1"
    )
    
    print("Running part2.infer()...")
    try:
        result = infer(bundle)
        print("\n--- Detection Result ---")
        print(json.dumps(result, indent=2))
        print("\nSuccess!")
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
