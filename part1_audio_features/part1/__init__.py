from typing import Optional
import os

from . import io, preprocess, features_acoustic, features_deep, bundle, config, utils

def extract_features(audio_base64: str, language_hint: Optional[str] = None) -> bundle.FeatureBundle:
    """
    Main pipeline function.
    
    Args:
        audio_base64: Base64 encoded MP3 string.
        language_hint: Optional language code (not used in Part 1 logic but passed for API compliance).
        
    Returns:
        FeatureBundle object.
    """
    wav_path = None
    try:
        # 1. Decode & Validate
        wav_path, metadata = io.decode_and_validate(audio_base64)
        
        # 2. Preprocess
        waveform = preprocess.preprocess_audio(wav_path)
        
        # 3. Acoustic Features
        acoustic = features_acoustic.extract_acoustic_features(waveform, sr=config.SAMPLE_RATE)
        
        # 4. Deep Embeddings
        if config.USE_DEEP_FEATURES:
            embeddings = features_deep.extract_deep_embeddings(waveform, sr=config.SAMPLE_RATE)
        else:
            # Return dummy embeddings to maintain schema compatibility
            embeddings = np.zeros(1536, dtype=np.float32)
            utils.logger.debug("Skipping deep embeddings (disabled in config)")
        
        # 5. Bundle
        feat_bundle = bundle.FeatureBundle(
            acoustic_features=acoustic,
            deep_embeddings=embeddings,
            metadata=metadata,
            version=config.BUNDLE_VERSION
        )



        return feat_bundle
        
    except Exception as e:
        utils.logger.error(f"Pipeline failed: {e}")
        raise e
    finally:
        # Cleanup processed wav file if it exists (io.py might have already cleaned temp mp3)
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
