import pytest
import torch
import numpy as np
from part2 import model, config

def test_model_output_shape():
    input_dim = config.INPUT_DIM_DEFAULT
    batch_size = 5
    
    clf = model.SimpleClassifier(input_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    
    output = clf(dummy_input)
    assert output.shape == (batch_size,)
    assert output.dtype == torch.float32

def test_config_consistency():
    assert len(config.HIDDEN_LAYERS) == 3
    assert config.DEFAULT_THRESHOLD == 0.5
