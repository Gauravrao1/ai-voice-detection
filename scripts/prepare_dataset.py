"""
Prepare dataset for training
Organizes audio files and creates train/test splits
"""

import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

def organize_dataset(data_dir="data", output_file="dataset_manifest.json"):
    """
    Organize audio files into a structured manifest
    """
    manifest = {
        "train": [],
        "test": []
    }
    
    # Collect human voices
    human_dir = Path(data_dir) / "common_voice"
    if human_dir.exists():
        for audio_file in human_dir.rglob("*.mp3"):
            manifest["train"].append({
                "file": str(audio_file),
                "label": 0,  # HUMAN
                "language": audio_file.parent.name
            })
    
    # Collect AI-generated voices
    ai_dir = Path(data_dir) / "ai_generated"
    if ai_dir.exists():
        for audio_file in ai_dir.rglob("*.mp3"):
            manifest["train"].append({
                "file": str(audio_file),
                "label": 1,  # AI_GENERATED
                "language": audio_file.parent.name
            })
    
    # Split train/test
    all_samples = manifest["train"]
    train_samples, test_samples = train_test_split(
        all_samples, 
        test_size=0.2, 
        random_state=42,
        stratify=[s["label"] for s in all_samples]
    )
    
    manifest["train"] = train_samples
    manifest["test"] = test_samples
    
    # Save manifest
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Dataset manifest created: {output_file}")
    logger.info(f"   Train samples: {len(train_samples)}")
    logger.info(f"   Test samples: {len(test_samples)}")
    
    return manifest

if __name__ == "__main__":
    organize_dataset()