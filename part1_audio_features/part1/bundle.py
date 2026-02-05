from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import json

@dataclass
class FeatureBundle:
    acoustic_features: Dict[str, float]
    deep_embeddings: np.ndarray  # float32 array
    metadata: Dict[str, Any]
    version: str = "part1-v1"

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary (excluding huge embeddings)."""
        return {
            "acoustic_features": self.acoustic_features,
            "metadata": self.metadata,
            "version": self.version,
            "deep_embeddings_shape": list(self.deep_embeddings.shape)
        }

    def save_npz(self, path: str):
        """Saves embeddings and metadata to npz."""
        np.savez_compressed(
            path,
            embeddings=self.deep_embeddings,
            acoustic=json.dumps(self.acoustic_features),
            metadata=json.dumps(self.metadata),
            version=self.version
        )
