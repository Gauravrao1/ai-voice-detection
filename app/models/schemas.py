from pydantic import BaseModel, Field, validator
from typing import Literal
from app.config import settings

class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio"
    )
    audioFormat: str = Field(
        default="mp3",
        description="Audio format (must be mp3)"
    )
    audioBase64: str = Field(
        ...,
        description="Base64 encoded audio file",
        min_length=100
    )
    
    @validator('audioFormat')
    def validate_audio_format(cls, v):
        if v.lower() != 'mp3':
            raise ValueError('Only mp3 format is supported')
        return v.lower()
    
    @validator('language')
    def validate_language(cls, v):
        if v not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(f'Language must be one of {settings.SUPPORTED_LANGUAGES}')
        return v

class VoiceDetectionResponse(BaseModel):
    status: Literal["success", "error"]
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.91,
                "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
            }
        }

class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        }