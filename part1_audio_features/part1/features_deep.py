import numpy as np
import logging
from . import config, utils

# Global model cache to avoid reloading on every call
_PROCESSOR = None
_MODEL = None

def load_model():
    """Loads model lazily."""
    global _PROCESSOR, _MODEL
    if _MODEL is None:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        utils.logger.info("Loading Wav2Vec2 model...")
        try:
            # Using facebook/wav2vec2-base-960h or just base which is smaller
            # Prompt suggested: facebook/wav2vec2-base
            model_name = "facebook/wav2vec2-base" 
            _PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
            _MODEL = Wav2Vec2Model.from_pretrained(model_name)
            
            # Apply dynamic quantization to reduce memory footprint (from ~360MB to ~180MB)
            # This is crucial for running on memory-constrained environments like Render Free tier
            _MODEL = torch.quantization.quantize_dynamic(
                _MODEL, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            _MODEL.eval()  # Set to eval mode
            utils.logger.info("Wav2Vec2 model loaded and quantized.")
        except Exception as e:
            utils.logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise e

def extract_deep_embeddings(waveform: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Extracts embeddings using Wav2Vec2.
    Returns: 1D numpy array (mean pooled + std pooled), or just mean.
    """
    load_model()
    
    try:
        # Normalize inputs for Wav2Vec2 (it expects raw speech input)
        # Processor handles padding and normalization
        inputs = _PROCESSOR(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = _MODEL(inputs.input_values)
            
        # Last hidden state: (Batch, Time, Dim) -> (1, T, 768)
        hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        # Pooling: Mean + Std to capture temporal dynamics
        # Validated against prompt suggestion: "Mean pooling across time... Or concatenation of mean + std"
        embedding_mean = np.mean(hidden_states, axis=0)
        embedding_std = np.std(hidden_states, axis=0)
        
        # Concatenate: 768 + 768 = 1536 dims
        final_embedding = np.concatenate([embedding_mean, embedding_std])
        
        return final_embedding.astype(np.float32)

    except Exception as e:
        utils.logger.error(f"Deep embedding extraction failed: {e}")
        # Return zeros or raise? Raising is safer for consistency.
        raise e
