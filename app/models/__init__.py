"""
Models module
Contains all detection models and schemas
"""

from app.models.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionResponse,
    ErrorResponse
)
from app.models.hf_detector import HuggingFaceDetector

__all__ = [
    "VoiceDetectionRequest",
    "VoiceDetectionResponse",
    "ErrorResponse",
    "HuggingFaceDetector",
]
