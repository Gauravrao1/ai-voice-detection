"""
Fine-tune Wav2Vec2 model for AI voice detection
"""

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Audio
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
import os

class VoiceDataset(Dataset):
    """
    Custom dataset for voice samples
    """
    def __init__(self, audio_files, labels, processor, max_length=160000):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio, sr = torchaudio.load(self.audio_files[idx])
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio = audio.squeeze().numpy()
        
        # Process
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class VoiceClassifier(torch.nn.Module):
    """
    Classifier on top of Wav2Vec2
    """
    def __init__(self, base_model):
        super().__init__()
        self.wav2vec2 = base_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 2)  # Binary classification
        )
    
    def forward(self, input_values):
        # Extract features
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling
        pooled = torch.mean(hidden_states, dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        return logits

def compute_metrics(pred):
    """
    Compute metrics for evaluation
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1
    }

def finetune_model(
    train_audio_files,
    train_labels,
    val_audio_files,
    val_labels,
    output_dir="ml_model/saved_models/finetuned_voice_detector",
    epochs=10,
    batch_size=8
):
    """
    Fine-tune Wav2Vec2 model
    """
    logger.info("🚀 Starting fine-tuning...")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load processor and base model
    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    base_model = Wav2Vec2Model.from_pretrained(model_name)
    
    # Create classifier
    model = VoiceClassifier(base_model).to(device)
    
    # Create datasets
    train_dataset = VoiceDataset(train_audio_files, train_labels, processor)
    val_dataset = VoiceDataset(val_audio_files, val_labels, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Training started...")
    trainer.train()
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.classifier, f"{output_dir}/classifier.pt")
    logger.info(f"✅ Model saved to {output_dir}")
    
    # Evaluate
    results = trainer.evaluate()
    logger.info(f"Evaluation Results: {results}")
    
    return model

# Example usage
if __name__ == "__main__":
    # You need to prepare your dataset
    # Example:
    # train_audio_files = ["path/to/audio1.mp3", "path/to/audio2.mp3", ...]
    # train_labels = [0, 1, 0, 1, ...]  # 0=HUMAN, 1=AI_GENERATED
    
    logger.info("⚠️  Please prepare your dataset first!")
    logger.info("Format:")
    logger.info("  train_audio_files = ['path/to/human1.mp3', 'path/to/ai1.mp3', ...]")
    logger.info("  train_labels = [0, 1, 0, 1, ...]  # 0=HUMAN, 1=AI")
    
    # Uncomment and modify when you have data:
    # finetune_model(
    #     train_audio_files=train_audio_files,
    #     train_labels=train_labels,
    #     val_audio_files=val_audio_files,
    #     val_labels=val_labels
    # )