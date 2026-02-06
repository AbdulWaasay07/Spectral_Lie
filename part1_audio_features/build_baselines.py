import os
import glob
import json
import numpy as np
import argparse
from tqdm import tqdm
from part1 import preprocess, features_acoustic, config

def compute_baselines(data_dir: str, output_path: str):
    """
    Iterates over WAV files in data_dir, computes acoustic features, 
    and aggregates statistics to produce a baseline JSON.
    """
    wav_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    if not wav_files:
        print(f"No WAV files found in {data_dir}")
        return

    aggregated = {}
    
    print(f"Processing {len(wav_files)} files...")
    for wav_path in tqdm(wav_files):
        try:
            # Preprocess & Extract
            # We skip io.decode checks and assume valid WAVs for baseline building
            waveform = preprocess.preprocess_audio(wav_path)
            feats = features_acoustic.extract_acoustic_features(waveform)
            
            for k, v in feats.items():
                if v is None or np.isnan(v):
                    continue
                if k not in aggregated:
                    aggregated[k] = []
                aggregated[k].append(v)
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue

    # Compute Stats
    baseline = {}
    for k, values in aggregated.items():
        if not values:
            continue
        vals = np.array(values)
        baseline[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75))
        }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing human speech WAVs")
    parser.add_argument("--output", type=str, default="baselines/human_baseline.json")
    args = parser.parse_args()
    
    compute_baselines(args.data_dir, args.output)
