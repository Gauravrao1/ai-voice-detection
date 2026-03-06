import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import Dict, Tuple, List
from loguru import logger
from app.config import settings
from app.utils.audio_processor import AudioProcessor
from concurrent.futures import ThreadPoolExecutor
import asyncio

class HuggingFaceDetector:
    """
    Hugging Face based AI Voice Detector
    Uses pre-trained models for better accuracy
    Optimized for faster processing with chunked inference
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.audio_processor = AudioProcessor()
        self.feature_extractor = None
        self.model = None
        self.use_half = settings.USE_HALF_PRECISION and self.device == "cuda"
        
        self._load_model()
    
    def _load_model(self):
        """
        Load Hugging Face model with optimizations
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
            
            # Enable half precision for faster GPU inference
            if self.use_half:
                self.model = self.model.half()
                logger.info("Using FP16 half precision for faster inference")
            
            # Set model to eval mode and enable inference optimizations
            self.model.eval()
            if hasattr(torch, 'inference_mode'):
                logger.info("PyTorch inference optimizations enabled")
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model Labels: {self.model.config.id2label}")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {str(e)}")
            raise
    
    def _process_chunk(self, audio_chunk: np.ndarray, target_sr: int) -> Tuple[int, float, dict]:
        """
        Process a single audio chunk and return prediction with all class probabilities
        """
        inputs = self.feature_extractor(
            audio_chunk,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        if self.use_half:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
            # Get all probabilities for each label
            all_probs = {self.model.config.id2label[i]: probabilities[0][i].item() 
                        for i in range(len(self.model.config.id2label))}
        
        return predicted_id, confidence, all_probs
    
    def _chunk_audio(self, audio: np.ndarray, sr: int, chunk_duration: float) -> List[np.ndarray]:
        """
        Split audio into chunks for faster processing
        """
        chunk_size = int(sr * chunk_duration)
        chunks = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) >= sr * 0.5:  # Only process chunks >= 0.5 seconds
                chunks.append(chunk)
        
        return chunks if chunks else [audio]
    
    def detect(self, audio_base64: str, language: str) -> Dict[str, any]:
        """
        Main detection function using Hugging Face model
        Optimized with chunked processing for large files
        """
        try:
            logger.info(f"HF Detection started for language: {language}")
            
            # Decode audio
            audio_bytes = self.audio_processor.decode_base64_audio(audio_base64)
            
            # Load audio
            target_sr = self.feature_extractor.sampling_rate if self.feature_extractor else 16000
            audio, sr = self.audio_processor.load_audio(audio_bytes, sr=target_sr)
            
            # Validate
            self.audio_processor.validate_audio(audio)
            
            # Determine if chunked processing is needed (for files > 30 seconds)
            duration = len(audio) / target_sr
            logger.info(f"Audio duration: {duration:.2f}s")
            
            if duration > settings.CHUNK_DURATION_SEC:
                # Use chunked processing for long audio
                chunks = self._chunk_audio(audio, target_sr, settings.CHUNK_DURATION_SEC)
                logger.info(f"Processing {len(chunks)} chunks for faster inference")
                
                # Process chunks and aggregate results
                predictions = []
                confidences = []
                all_probs_list = []
                
                for i, chunk in enumerate(chunks):
                    pred_id, conf, probs = self._process_chunk(chunk, target_sr)
                    predictions.append(pred_id)
                    confidences.append(conf)
                    all_probs_list.append(probs)
                    logger.debug(f"Chunk {i+1}/{len(chunks)}: pred={pred_id}, conf={conf:.3f}, probs={probs}")
                
                # Aggregate: use majority vote weighted by confidence
                from collections import Counter
                weighted_votes = Counter()
                for pred_id, conf in zip(predictions, confidences):
                    weighted_votes[pred_id] += conf
                
                predicted_id = weighted_votes.most_common(1)[0][0]
                confidence = sum(c for p, c in zip(predictions, confidences) if p == predicted_id) / len([p for p in predictions if p == predicted_id])
                # Average probabilities across chunks
                avg_probs = {}
                for key in all_probs_list[0].keys():
                    avg_probs[key] = sum(p[key] for p in all_probs_list) / len(all_probs_list)
            else:
                # Process entire audio at once for short files
                predicted_id, confidence, avg_probs = self._process_chunk(audio, target_sr)
                
            # Map label
            label = self.model.config.id2label[predicted_id]
            logger.info(f"Predicted Label: {label}, Confidence: {confidence}, All Probs: {avg_probs}")
            
            # ===== SPECTRAL ANALYSIS (Secondary Detection) =====
            spectral_features = self.audio_processor.analyze_spectral_features(audio, target_sr)
            spectral_ai_score = self.audio_processor.compute_ai_score(spectral_features)
            logger.info(f"Spectral AI Score: {spectral_ai_score:.2%}")
            logger.debug(f"Spectral Features: {spectral_features}")
            
            # Model-agnostic AI detection logic
            ai_keywords = ["fake", "spoof", "ai", "generated", "synthetic", "deepfake", "ai-generated", "ai_generated"]
            human_keywords = ["real", "human", "genuine", "authentic", "bonafide", "natural"]
            
            ai_prob = 0.0
            human_prob = 0.0
            label_lower = label.lower()
            
            for lbl, prob in avg_probs.items():
                lbl_lower = lbl.lower()
                if any(kw in lbl_lower for kw in ai_keywords):
                    ai_prob += prob
                elif any(kw in lbl_lower for kw in human_keywords):
                    human_prob += prob
            
            logger.info(f"Model AI prob: {ai_prob:.4f}, Human prob: {human_prob:.4f}")
            
            is_ai_label = any(kw in label_lower for kw in ai_keywords)
            
            # ===== ENHANCED MULTI-SIGNAL DETECTION =====
            # Modern AI voices are very good - we need aggressive detection
            ai_signals = []
            
            # Signal 1: Model directly says AI/fake
            if is_ai_label:
                ai_signals.append(f"model={label}")
            
            # Signal 2: High AI probability from model
            if ai_prob >= 0.08:  # Lowered threshold
                ai_signals.append(f"ai_prob={ai_prob:.0%}")
            
            # Signal 3: Model uncertainty (not very confident about human)
            if human_prob < 0.97:
                ai_signals.append(f"model_uncertain={human_prob:.0%}")
            
            # Signal 4: Spectral analysis
            if spectral_ai_score >= 0.25:  # Lowered threshold
                ai_signals.append(f"spectral={spectral_ai_score:.0%}")
            
            # Signal 5: Individual spectral features
            pitch_cv = spectral_features.get('pitch_cv', 0.2)
            jitter = spectral_features.get('jitter', 0.015)
            mfcc_delta_var = spectral_features.get('mfcc_delta_var', 15)
            mfcc_var = spectral_features.get('mfcc_var', 60)
            harmonic_ratio = spectral_features.get('harmonic_ratio', 0.75)
            rms_cv = spectral_features.get('rms_cv', 0.4)
            
            if pitch_cv < 0.18:
                ai_signals.append(f"pitch_cv={pitch_cv:.3f}")
            if jitter < 0.012:
                ai_signals.append(f"jitter={jitter:.4f}")
            if mfcc_delta_var < 12:
                ai_signals.append(f"mfcc_delta={mfcc_delta_var:.2f}")
            if mfcc_var < 50:
                ai_signals.append(f"mfcc_var={mfcc_var:.2f}")
            if harmonic_ratio > 0.88:
                ai_signals.append(f"harmonic={harmonic_ratio:.3f}")
            if rms_cv < 0.4:
                ai_signals.append(f"rms_cv={rms_cv:.3f}")
            
            logger.info(f"AI Signals ({len(ai_signals)}): {ai_signals}")
            
            # ===== FINAL CLASSIFICATION =====
            # Combined scoring with aggressive weights
            
            # Base combined score
            combined_ai_score = (
                0.25 * ai_prob +                    # Model AI probability
                0.45 * spectral_ai_score +          # Spectral analysis (primary)
                0.15 * (1 - human_prob) +           # Inverse of human probability
                0.15 * (len(ai_signals) / 10)       # Number of signals
            )
            
            # Boost for model AI detection
            if is_ai_label and confidence > 0.5:
                combined_ai_score = max(combined_ai_score, 0.75)
            
            # Boost for strong spectral detection
            if spectral_ai_score >= 0.50:
                combined_ai_score = max(combined_ai_score, 0.65)
            elif spectral_ai_score >= 0.35:
                combined_ai_score = max(combined_ai_score, 0.50)
            
            # Boost for multiple signals (collective evidence)
            if len(ai_signals) >= 6:
                combined_ai_score = max(combined_ai_score, 0.70)
            elif len(ai_signals) >= 4:
                combined_ai_score = max(combined_ai_score, 0.55)
            elif len(ai_signals) >= 3:
                combined_ai_score = max(combined_ai_score, 0.45)
            
            logger.info(f"Combined AI Score: {combined_ai_score:.2%}")
            
            # Classification thresholds (aggressive)
            if combined_ai_score >= 0.35:
                classification = "AI_GENERATED"
                final_confidence = min(0.99, 0.5 + combined_ai_score * 0.5)
                
                if is_ai_label:
                    explanation = f"Model detected AI ({label}, {confidence:.0%})"
                elif spectral_ai_score >= 0.4:
                    explanation = f"Spectral analysis: {spectral_ai_score:.0%} synthetic patterns"
                elif len(ai_signals) >= 4:
                    explanation = f"Multiple AI indicators: {', '.join(ai_signals[:4])}"
                else:
                    explanation = f"AI patterns detected (score: {combined_ai_score:.0%})"
                    
            elif len(ai_signals) >= 2 and (spectral_ai_score >= 0.20 or human_prob < 0.95):
                classification = "AI_GENERATED"
                final_confidence = 0.55 + combined_ai_score * 0.3
                explanation = f"Suspicious patterns: {', '.join(ai_signals[:3])}"
                
            else:
                classification = "HUMAN"
                final_confidence = max(human_prob, 1 - combined_ai_score)
                explanation = f"Natural voice (human_prob={human_prob:.0%}, spectral_human={1-spectral_ai_score:.0%})"
            
            return {
                "classification": classification,
                "confidence": round(float(final_confidence), 2),
                "explanation": explanation,
                "details": {
                    "model_label": label,
                    "model_confidence": round(confidence, 3),
                    "ai_probability": round(ai_prob, 3),
                    "human_probability": round(human_prob, 3),
                    "spectral_ai_score": round(spectral_ai_score, 3),
                    "combined_score": round(combined_ai_score, 3),
                    "signals_count": len(ai_signals),
                    "signals": ai_signals
                }
            }
        
        except Exception as e:
            logger.error(f"HF Detection error: {str(e)}")
            raise
