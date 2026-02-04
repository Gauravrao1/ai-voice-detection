import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import Dict, Tuple
from loguru import logger
from app.config import settings
from app.utils.audio_processor import AudioProcessor

class HuggingFaceDetector:
    """
    Hugging Face based AI Voice Detector
    Uses pre-trained models for better accuracy
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.audio_processor = AudioProcessor()
        self.feature_extractor = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """
        Load Hugging Face model
        """
        try:
            logger.info(f"Loading Hugging Face model: {settings.HF_MODEL_NAME}...")
            
            # Load feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                settings.HF_MODEL_NAME,
                cache_dir=settings.HF_MODEL_CACHE_DIR
            )
            
            self.model = AutoModelForAudioClassification.from_pretrained(
                settings.HF_MODEL_NAME,
                cache_dir=settings.HF_MODEL_CACHE_DIR
            ).to(self.device)
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model Labels: {self.model.config.id2label}")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {str(e)}")
            raise
    
    def detect(self, audio_base64: str, language: str) -> Dict[str, any]:
        """
        Main detection function using Hugging Face model
        """
        try:
            logger.info(f"HF Detection started for language: {language}")
            
            # Decode audio
            audio_bytes = self.audio_processor.decode_base64_audio(audio_base64)
            
            # Load audio
            # Note: feature extractor usually expects 16000Hz
            target_sr = self.feature_extractor.sampling_rate if self.feature_extractor else 16000
            audio, sr = self.audio_processor.load_audio(audio_bytes, sr=target_sr)
            
            # Validate
            self.audio_processor.validate_audio(audio)
            
            # Prepare inputs
            inputs = self.feature_extractor(
                audio,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
                
            # Map label
            label = self.model.config.id2label[predicted_id]
            logger.info(f"Predicted Label: {label}, Confidence: {confidence}")
            
            # Determine classification with calibrated thresholds per language
            # Common labels: "fake", "spoof", "AI" vs "real", "human", "bonafide"
            label_lower = label.lower()
            lang_threshold = settings.LANGUAGE_THRESHOLDS.get(language, settings.CONFIDENCE_THRESHOLD)
            is_ai_label = any(x in label_lower for x in ["fake", "spoof", "ai", "generated"])
            if is_ai_label and confidence >= lang_threshold:
                classification = "AI_GENERATED"
                explanation = f"AI patterns detected ({label}); confidence {confidence:.2f} >= threshold {lang_threshold:.2f}"
            else:
                classification = "HUMAN"
                if is_ai_label:
                    explanation = f"Model suggested '{label}', but confidence {confidence:.2f} < threshold {lang_threshold:.2f} — treated as human"
                else:
                    explanation = f"Natural voice patterns detected ({label})"
            
            return {
                "classification": classification,
                "confidence": round(float(confidence), 2),
                "explanation": explanation
            }
        
        except Exception as e:
            logger.error(f"HF Detection error: {str(e)}")
            raise
