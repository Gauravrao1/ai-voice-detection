"""
Core functionality module
Contains authentication, exceptions, and other core utilities
"""

from app.core.auth import verify_api_key
from app.core.exceptions import (
    AudioProcessingError,
    InvalidAudioFormatError,
    AudioTooLargeError,
    ModelNotFoundError
)

__all__ = [
    "verify_api_key",
    "AudioProcessingError",
    "InvalidAudioFormatError",
    "AudioTooLargeError",
    "ModelNotFoundError",
]