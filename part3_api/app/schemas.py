from pydantic import BaseModel, Field
from typing import Optional

class DetectRequest(BaseModel):
    audioBase64: str = Field(
        ..., 
        description="The base64 encoded audio data.",
        example="SUQzBAAAAAAAI1..."
    )
    language: str = Field(
        ..., 
        description="The language of the audio (e.g., 'English', Hindi').",
        example="English"
    )
    audioFormat: str = Field(
        "mp3",
        description="The format of the audio (e.g., 'mp3', 'wav').",
        example="mp3"
    )

class DetectResponse(BaseModel):
    classification: str = Field(..., description="Prediction: 'Human' or 'AI-Generated'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    explanation: str = Field(..., description="Human-readable explanation of the decision")
    model_version: str = Field("v1.0", description="Version of the detection model")
    request_id: str = Field(..., description="Unique ID for this request")
