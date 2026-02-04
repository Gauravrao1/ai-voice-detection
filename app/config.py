from pydantic_settings import BaseSettings
from typing import Literal
from loguru import logger

class Settings(BaseSettings):
    # API Settings
    API_KEY: str = "sk_test_123456789"
    ENVIRONMENT: Literal["development", "production"] = "development"
    
    # Model Settings
    MODEL_TYPE: Literal["huggingface"] = "huggingface"
    
    # Hugging Face Model Settings
    HF_MODEL_NAME: str = "MelodyMachine/Deepfake-audio-detection"
    HF_MODEL_CACHE_DIR: str = "./models/huggingface_cache"
    USE_GPU: bool = True
    
    # Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.90
    LANGUAGE_THRESHOLDS: dict = {
        "English": 0.90,
        "Hindi": 0.92,
        "Tamil": 0.92,
        "Malayalam": 0.92,
        "Telugu": 0.92
    }
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Audio Settings
    MAX_AUDIO_SIZE_MB: int = 2
    SAMPLE_RATE: int = 16000
    
    # Supported Languages
    SUPPORTED_LANGUAGES: list = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
logger.info(f"DEBUG: Loaded settings. HF_MODEL_NAME={settings.HF_MODEL_NAME}")
