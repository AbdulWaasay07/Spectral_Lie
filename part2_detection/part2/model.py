import torch
import torch.nn as nn
from . import config

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Simplified architecture for 92 acoustic features to prevent saturation
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Logits output (batch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, input_dim)
        Returns:
            Logits of shape (batch, 1)
        """
        return self.net(x)
