from fastapi import HTTPException, status

class AudioProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing error: {detail}"
        )

class InvalidAudioFormatError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio format. Only MP3 is supported"
        )

class AudioTooLargeError(HTTPException):
    def __init__(self, max_size_mb: int):
        super().__Exception__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Maximum size: {max_size_mb}MB"
        )

class ModelNotFoundError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection model not loaded"
        )