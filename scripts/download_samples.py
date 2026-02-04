"""
Download sample audio datasets for training
"""

from datasets import load_dataset
import os
from loguru import logger

def download_common_voice():
    """
    Download Common Voice dataset (for HUMAN voices)
    """
    logger.info("Downloading Common Voice dataset...")
    
    # Download specific languages
    languages = ['ta', 'en', 'hi', 'ml', 'te']  # Tamil, English, Hindi, Malayalam, Telugu
    
    for lang in languages:
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_13_0",
                lang,
                split="train[:100]",  # Small sample
                cache_dir="./data/common_voice"
            )
            logger.info(f"✅ Downloaded {lang}: {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to download {lang}: {e}")

def download_ai_generated_samples():
    """
    You'll need to generate or find AI-generated samples
    Options:
    1. Use TTS services (ElevenLabs, Google TTS, etc.)
    2. Use local TTS models (Coqui TTS, ESPnet)
    3. Find existing deepfake audio datasets
    """
    logger.info("ℹ️  For AI-generated samples, you need to:")
    logger.info("1. Use TTS services to generate synthetic voices")
    logger.info("2. Or download from deepfake audio datasets")
    logger.info("3. Save them in data/ai_generated/")

if __name__ == "__main__":
    # Create directories
    os.makedirs("./data/common_voice", exist_ok=True)
    os.makedirs("./data/ai_generated", exist_ok=True)
    
    # Download human voices
    download_common_voice()
    
    # Guide for AI samples
    download_ai_generated_samples()