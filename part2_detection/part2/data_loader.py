import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from . import utils, config

class FeatureDataset(Dataset):
    def __init__(self, data_dir: str, labels_file: str):
        """
        Args:
            data_dir: Directory containing .npz feature files.
            labels_file: CSV or JSON mapping filename -> label (0=human, 1=AI).
        """
        self.data_dir = data_dir
        with open(labels_file, "r") as f:
            # Expecting simple dict "filename": label
            self.labels_map = json.load(f)
        self.samples = list(self.labels_map.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        label = self.labels_map[filename]
        
        path = os.path.join(self.data_dir, filename)
        
        # Load NPZ
        # We need to reconstruct a FeatureBundle-like object to use utils.prepare_input
        # Or just manually reuse logic.
        data = np.load(path, allow_pickle=True)
        
        # Mock object to satisfy utils.run_input logic
        from types import SimpleNamespace
        bundle = SimpleNamespace(
            deep_embeddings=data["embeddings"],
            acoustic_features=json.loads(str(data["acoustic"]))
        )
        
        # Reuse utils logic for consistency (normalization, concatenation)
        x = utils.prepare_input(bundle).squeeze(0) # Remove batch dim
        y = torch.tensor(label).float()
        
        return x, y

def get_dataloader(data_dir: str, labels_file: str, batch_size: int = 32):
    dataset = FeatureDataset(data_dir, labels_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
