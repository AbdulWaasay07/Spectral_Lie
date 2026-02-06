import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaler(nn.Module):
    """
    Scales logits using a learnable single scalar temperature (T > 0)
    and a bias term (Platt Scaling: a*x + b).
    Probability = sigmoid((logit / T) + bias)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits / self.temperature) + self.bias

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, epochs: int = 100):
        """
        Tunes temperature and bias on validation set (logits, labels).
        """
        optimizer = optim.LBFGS([self.temperature, self.bias], lr=lr, max_iter=epochs)
        criterion = nn.BCEWithLogitsLoss()
        
        logits = logits.detach()
        labels = labels.float()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        # We clamp temperature (scale) to keep it positive and sane
        with torch.no_grad():
            self.temperature.clamp_(min=0.1, max=10.0)
            
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probability 0.0-1.0"""
        scaled = self.forward(logits)
        return torch.sigmoid(scaled)
