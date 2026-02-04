import base64
import io
import librosa
import numpy as np
from typing import Tuple
from app.config import settings
from app.core.exceptions import AudioProcessingError, AudioTooLargeError

class AudioProcessor:
    """
    Audio processing utilities
    """
    
    @staticmethod
    def decode_base64_audio(audio_base64: str) -> bytes:
        """
        Decode base64 encoded audio
        """
        try:
            # Remove data URI prefix if present
            if ',' in audio_base64:
                audio_base64 = audio_base64.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_base64)
            
            # Check file size
            size_mb = len(audio_bytes) / (1024 * 1024)
            if size_mb > settings.MAX_AUDIO_SIZE_MB:
                raise AudioTooLargeError(settings.MAX_AUDIO_SIZE_MB)
            
            return audio_bytes
        
        except base64.binascii.Error as e:
            raise AudioProcessingError(f"Invalid base64 encoding: {str(e)}")
    
    @staticmethod
    def load_audio(audio_bytes: bytes, sr: int = None) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes
        """
        try:
            if sr is None:
                sr = settings.SAMPLE_RATE
            
            audio, sample_rate = librosa.load(
                io.BytesIO(audio_bytes),
                sr=sr,
                mono=True
            )
            
            return audio, sample_rate
        
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    
    @staticmethod
    def validate_audio(audio: np.ndarray) -> bool:
        """
        Validate audio data
        """
        if len(audio) == 0:
            raise AudioProcessingError("Audio is empty")
        
        if len(audio) < settings.SAMPLE_RATE * 0.5:  # Minimum 0.5 seconds
            raise AudioProcessingError("Audio too short (minimum 0.5 seconds)")
        
        return True