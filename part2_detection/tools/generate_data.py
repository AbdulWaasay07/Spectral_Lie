import numpy as np
import os
import json
import uuid
import argparse

# Constants
LANGUAGES = ["English", "Hindi", "Malayalam", "Telugu", "Tamil"]
NUM_SAMPLES_PER_LANG = 1000  # 500 Human, 500 AI per lang
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def generate_sample(label: int, lang: str):
    """
    Generates a synthetic FeatureBundle dictionary.
    label: 0 = Human, 1 = AI
    
    Feature distributions are calibrated from real audio samples:
    - Real human audio has: mfcc_mean_0 ~ -305, mfcc_std_0 ~ 158, pitch_std ~ 81
    """
    acoustic = {}
    
    # === KEY DIFFERENTIATORS (HIGH OVERLAP for calibration) ===
    
    # 1. Jitter/Shimmer - Large overlap
    if label == 1:  # AI
        acoustic["jitter_local"] = np.random.uniform(0.010, 0.035) 
        acoustic["shimmer_local"] = np.random.uniform(0.04, 0.12)
    else:  # Human
        acoustic["jitter_local"] = np.random.uniform(0.015, 0.05)
        acoustic["shimmer_local"] = np.random.uniform(0.06, 0.18)
    
    # 2. HNR (Harmonics-to-Noise Ratio) - Massive overlap
    if label == 1:  # AI
        acoustic["hnr"] = np.random.uniform(8, 22)
    else:  # Human
        acoustic["hnr"] = np.random.uniform(5, 18)
    
    # 3. Pitch variability - Subtle difference
    acoustic["pitch_mean"] = np.random.normal(130, 40)
    if label == 1:  # AI
        acoustic["pitch_std"] = np.random.uniform(40, 85)
    else:  # Human
        acoustic["pitch_std"] = np.random.uniform(45, 120)
    
    # 4. Voiced ratio
    acoustic["voiced_ratio"] = np.random.uniform(0.3, 0.6)
    
    # === MFCC FEATURES (Environment Noise + Subtle Drift) ===
    mfcc_base_means = [-305, 55, 16, 12, 11, 2, 1, -19, -8, 0.3, -0.2, -5.5, -1.8]
    mfcc_base_stds = [158, 75, 33, 24, 17, 16, 13, 19, 11, 12, 9, 9, 10]
    
    env_shift = np.random.normal(0, 8, 13) 
    
    for i in range(13):
        # AI and Human share almost same means, only slight distributional drift
        drift = 0 if label == 0 else np.random.uniform(-3, 3)
        acoustic[f"mfcc_mean_{i}"] = np.random.normal(mfcc_base_means[i] + drift + env_shift[i], mfcc_base_stds[i] * 0.4)
        acoustic[f"mfcc_std_{i}"] = abs(np.random.normal(mfcc_base_stds[i], mfcc_base_stds[i] * 0.3))

    
    # Delta MFCCs (rate of change) - near zero mean, smaller std
    for i in range(13):
        acoustic[f"mfcc_delta_mean_{i}"] = np.random.normal(0, 1)
        acoustic[f"mfcc_delta_std_{i}"] = abs(np.random.normal(5, 3))
        acoustic[f"mfcc_delta2_mean_{i}"] = np.random.normal(0, 0.5)
        acoustic[f"mfcc_delta2_std_{i}"] = abs(np.random.normal(3, 2))
    
    # === SPECTRAL FEATURES ===
    acoustic["spectral_centroid_mean"] = np.random.normal(2700, 500)
    acoustic["spectral_centroid_std"] = np.random.normal(1700, 400)
    acoustic["spectral_rolloff_mean"] = np.random.normal(4700, 800)
    acoustic["spectral_rolloff_std"] = np.random.normal(2300, 500)
    acoustic["spectral_flatness_mean"] = np.random.uniform(0.05, 0.15)
    acoustic["spectral_flatness_std"] = np.random.uniform(0.08, 0.18)
    acoustic["zcr_mean"] = np.random.uniform(0.15, 0.35)
    acoustic["zcr_std"] = np.random.uniform(0.15, 0.30)

    # --- Deep Embeddings (1536 dim) ---
    # Centered differently for AI vs Human with significant overlap
    mu = 0.02 if label == 1 else -0.02
    sigma = 0.15
    embeddings = np.random.normal(mu, sigma, 1536).astype(np.float32)
    
    return acoustic, embeddings

def main():
    print(f"Generating synthetic data for {LANGUAGES}")
    
    train_dir = os.path.join(OUTPUT_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Create Labels file
    labels_map = {}
    
    files_created = 0
    
    for lang in LANGUAGES:
        print(f"  Processing {lang}...")
        for i in range(NUM_SAMPLES_PER_LANG):
            # 50/50 split
            label = 0 if i < (NUM_SAMPLES_PER_LANG // 2) else 1
            
            acoustic, emb = generate_sample(label, lang)
            
            # Metadata
            filename = f"{lang}_{uuid.uuid4().hex[:8]}.npz"
            path = os.path.join(train_dir, filename)
            
            # Save
            # We dump acoustic as JSON string inside NPZ to match how data_loader handles it?
            # Or data_loader expects 'acoustic' key in npz.
            # Let's save as dictionary in npz object array or json string. 
            # Looking at data_loader.py: json.loads(str(data["acoustic"])) implies it expects a string.
            np.savez_compressed(path, embeddings=emb, acoustic=json.dumps(acoustic))
            
            labels_map[filename] = label
            files_created += 1
            
    # Save train labels
    with open(os.path.join(train_dir, "labels.json"), "w") as f:
        json.dump(labels_map, f)
        
    print(f"Done! Created {files_created} samples in {train_dir}")
    print(f"Labels saved to {os.path.join(train_dir, 'labels.json')}")

    # Validation (20% size)
    val_dir = os.path.join(OUTPUT_DIR, "val")
    os.makedirs(val_dir, exist_ok=True)
    val_labels = {}
    print("Generating validation set...")
    for lang in LANGUAGES:
        for i in range(200): # 100 each
            label = 0 if i < 100 else 1
            acoustic, emb = generate_sample(label, lang)
            filename = f"val_{lang}_{uuid.uuid4().hex[:8]}.npz"
            np.savez_compressed(os.path.join(val_dir, filename), embeddings=emb, acoustic=json.dumps(acoustic))
            val_labels[filename] = label
            
    with open(os.path.join(val_dir, "labels.json"), "w") as f:
        json.dump(val_labels, f)

if __name__ == "__main__":
    main()
