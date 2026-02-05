from typing import Dict, Any, List

def generate_explanation(
    features: Dict[str, float], 
    baselines: Dict[str, Any], 
    confidence: float, 
    threshold: float = 0.5
) -> str:
    """
    Generates plain-English explanation based on rule checks and model confidence.
    """
    is_fake = confidence >= threshold
    cues = []

    # Safe get helper
    def get_val(feat_dict, key):
        return feat_dict.get(key, 0.0)

    # --- Rule Checks ---
    
    # 1. Stability (Jitter/Shimmer)
    # AI often has perfect stability (low jitter/shimmer) compared to humans
    jitter = get_val(features, "jitter_local")
    if baselines and "jitter_local" in baselines:
        human_median = baselines["jitter_local"]["median"]
        if jitter < (human_median * 0.5):
            cues.append("Unusually low jitter (robotic pitch stability)")

    # 2. Pitch Variance
    pitch_var = get_val(features, "pitch_std") ** 2  # Approx variance
    if baselines and "pitch_mean" in baselines:
        # Assuming 'pitch_mean' baseline stores stats about frequency
        # Real baseline should have pitch_std stats
        pass 

    # 3. Spectral cues
    hnr = get_val(features, "hnr")
    if hnr > 30.0:  # Arbitrary high HNR check
        cues.append("Extremely high harmonicity (clean synthesis)")

    # --- Construct Message ---
    
    # --- Construct Message ---
    
    is_ai = confidence >= threshold
    verdict_str = "AI-generated" if is_ai else "Human"
    
    # Use winning class confidence for display
    display_conf = confidence if is_ai else (1.0 - confidence)
    
    # Confidence adjective
    if display_conf < 0.6:
        conf_adj = "uncertain"
    elif display_conf > 0.85:
        conf_adj = "highly confident"
    else:
        conf_adj = "moderate"

    base_msg = f"The voice is classified as {verdict_str} with {conf_adj} probability ({display_conf:.2f})."
    
    if cues and is_ai:
        reasoning = " Signals include: " + "; ".join(cues[:3]) + "."
    elif not is_ai:
        reasoning = " Acoustic features align with human speech patterns."
    else:
        reasoning = " Relying on deep feature analysis."

    return base_msg + reasoning
