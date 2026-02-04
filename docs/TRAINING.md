# Training Guide

## Step 1: Prepare Data

1. Download human voices:
```bash
python scripts/download_samples.py
```

2. Generate AI voices using TTS services

3. Organize:
```
data/
  common_voice/
    tamil/*.mp3
    english/*.mp3
  ai_generated/
    tamil/*.mp3
    english/*.mp3
```

## Step 2: Train Traditional Model
```bash
python ml_model/train_model.py
```

## Step 3: Fine-tune Hugging Face Model
```bash
python ml_model/finetune_model.py
```

## Step 4: Test
```bash
python scripts/test_api.py
```