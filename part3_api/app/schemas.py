from pydantic import BaseModel, Field
from typing import Optional

class DetectRequest(BaseModel):
    audio_base64: str = Field(
        ..., 
        description="Base64-encoded audio file (MP3/WAV)",
        example="SUQzBAAAAAAAI1..."
    )
    language: Optional[str] = Field(
        None, 
        description="Optional language hint (English, Hindi, etc.)",
        example="English"
    )

class DetectResponse(BaseModel):
    classification: str = Field(..., description="Prediction: 'Human' or 'AI-Generated'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    explanation: str = Field(..., description="Human-readable explanation of the decision")
    model_version: str = Field("v1.0", description="Version of the detection model")
    request_id: str = Field(..., description="Unique ID for this request")
