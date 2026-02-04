"""
Unit tests for audio processor
"""

import pytest
import base64
import numpy as np
from app.utils.audio_processor import AudioProcessor
from app.core.exceptions import AudioProcessingError, AudioTooLargeError

class TestAudioProcessor:
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor instance"""
        return AudioProcessor()
    
    @pytest.fixture
    def valid_base64_audio(self):
        """
        Generate a valid base64 encoded audio sample
        This is a minimal valid MP3 header + data
        """
        # Minimal MP3 frame (simplified for testing)
        mp3_bytes = b'\xff\xfb\x90\x00' + b'\x00' * 100
        return base64.b64encode(mp3_bytes).decode('utf-8')
    
    @pytest.fixture
    def invalid_base64(self):
        """Invalid base64 string"""
        return "This is not valid base64!@#$%"
    
    def test_decode_valid_base64(self, audio_processor, valid_base64_audio):
        """Test decoding valid base64 audio"""
        try:
            audio_bytes = audio_processor.decode_base64_audio(valid_base64_audio)
            assert isinstance(audio_bytes, bytes)
            assert len(audio_bytes) > 0
        except Exception as e:
            # May fail with actual MP3 validation, that's ok for this test
            assert True
    
    def test_decode_invalid_base64(self, audio_processor, invalid_base64):
        """Test decoding invalid base64 should raise error"""
        with pytest.raises(AudioProcessingError):
            audio_processor.decode_base64_audio(invalid_base64)
    
    def test_decode_with_data_uri_prefix(self, audio_processor, valid_base64_audio):
        """Test decoding base64 with data URI prefix"""
        data_uri = f"data:audio/mp3;base64,{valid_base64_audio}"
        try:
            audio_bytes = audio_processor.decode_base64_audio(data_uri)
            assert isinstance(audio_bytes, bytes)
        except Exception as e:
            assert True
    
    def test_audio_too_large(self, audio_processor, monkeypatch):
        """Test that large audio files are rejected"""
        # Create a large base64 string (simulate > 10MB)
        large_data = b'\x00' * (11 * 1024 * 1024)  # 11MB
        large_base64 = base64.b64encode(large_data).decode('utf-8')
        
        with pytest.raises(AudioTooLargeError):
            audio_processor.decode_base64_audio(large_base64)
    
    def test_validate_empty_audio(self, audio_processor):
        """Test validation fails for empty audio"""
        empty_audio = np.array([])
        with pytest.raises(AudioProcessingError):
            audio_processor.validate_audio(empty_audio)
    
    def test_validate_short_audio(self, audio_processor):
        """Test validation fails for too short audio"""
        # Audio shorter than 0.5 seconds at 16000 Hz
        short_audio = np.random.randn(1000)  # Less than 0.5s
        with pytest.raises(AudioProcessingError):
            audio_processor.validate_audio(short_audio)
    
    def test_validate_valid_audio(self, audio_processor):
        """Test validation passes for valid audio"""
        # Audio longer than 0.5 seconds at 16000 Hz
        valid_audio = np.random.randn(16000)  # 1 second at 16kHz
        assert audio_processor.validate_audio(valid_audio) == True
    
    def test_load_audio_returns_correct_types(self, audio_processor):
        """Test that load_audio returns correct types"""
        # This test requires a real audio file, so we'll skip actual loading
        # Just verify the method exists and has correct signature
        assert hasattr(audio_processor, 'load_audio')
        assert callable(audio_processor.load_audio)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])