import base64
import io
import librosa
import numpy as np
from typing import Tuple, Dict
from app.config import settings
from app.core.exceptions import AudioProcessingError, AudioTooLargeError

class AudioProcessor:
    """
    Audio processing utilities with AI voice detection features
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
    
    @staticmethod
    def analyze_spectral_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Advanced spectral analysis for AI voice detection.
        AI voices typically exhibit:
        - Uniform pitch (low variation)
        - Consistent spectral patterns
        - Unnatural harmonic structure
        - Missing micro-variations present in human speech
        - Artifacts in high frequencies
        """
        features = {}
        
        try:
            # ===== PITCH ANALYSIS =====
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=500)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 10:
                pitch_std = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
                features['pitch_cv'] = pitch_std / pitch_mean if pitch_mean > 0 else 0
                # Pitch range ratio - humans have more dynamic range
                pitch_range = (np.max(pitch_values) - np.min(pitch_values)) / pitch_mean
                features['pitch_range'] = pitch_range
                # Jitter approximation (pitch perturbation)
                pitch_diffs = np.abs(np.diff(pitch_values))
                features['jitter'] = np.mean(pitch_diffs) / pitch_mean if pitch_mean > 0 else 0
            else:
                features['pitch_cv'] = 0.05  # Suspicious if no pitch detected
                features['pitch_range'] = 0.1
                features['jitter'] = 0.001
            
            # ===== SPECTRAL FEATURES =====
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_cv'] = features['spectral_centroid_std'] / features['spectral_centroid_mean'] if features['spectral_centroid_mean'] > 0 else 0
            
            # Spectral bandwidth - AI often has narrower bandwidth
            spectral_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth_std'] = np.std(spectral_bw)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bw)
            
            # Spectral contrast - difference between peaks and valleys
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast_mean'] = np.mean(spectral_contrast)
            features['spectral_contrast_std'] = np.std(spectral_contrast)
            
            # Spectral rolloff - frequency below which 85% of energy is contained
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
            features['spectral_rolloff_std'] = np.std(rolloff)
            features['spectral_rolloff_mean'] = np.mean(rolloff)
            
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)
            
            # ===== TEMPORAL FEATURES =====
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_std'] = np.std(zcr)
            features['zcr_mean'] = np.mean(zcr)
            
            # RMS energy variation - humans have more dynamic volume
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_std'] = np.std(rms)
            features['rms_mean'] = np.mean(rms)
            features['rms_cv'] = features['rms_std'] / features['rms_mean'] if features['rms_mean'] > 0 else 0
            
            # ===== MFCC ANALYSIS =====
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            features['mfcc_var'] = np.mean(np.var(mfccs, axis=1))
            # Delta MFCCs - temporal changes, AI has less natural transitions
            mfcc_delta = librosa.feature.delta(mfccs)
            features['mfcc_delta_var'] = np.mean(np.var(mfcc_delta, axis=1))
            # Delta-delta MFCCs (acceleration)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features['mfcc_delta2_var'] = np.mean(np.var(mfcc_delta2, axis=1))
            
            # ===== HARMONIC ANALYSIS =====
            # Harmonic-to-noise approximation using harmonic/percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            features['harmonic_ratio'] = harmonic_energy / total_energy if total_energy > 0 else 0.5
            
            # Harmonics analysis - AI may have artificial harmonic patterns
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_std'] = np.mean(np.std(chroma, axis=1))
            
            # ===== SILENCE/PAUSE ANALYSIS =====
            # Natural speech has irregular pauses
            silent_frames = np.sum(rms < 0.01)
            features['silence_ratio'] = silent_frames / len(rms) if len(rms) > 0 else 0
            
            # ===== HIGH FREQUENCY ARTIFACTS =====
            # AI often has artifacts in higher frequencies
            stft = np.abs(librosa.stft(audio))
            n_bins = stft.shape[0]
            high_freq_bins = stft[int(n_bins * 0.7):, :]  # Top 30% frequencies
            low_freq_bins = stft[:int(n_bins * 0.3), :]   # Bottom 30% frequencies
            high_energy = np.mean(high_freq_bins)
            low_energy = np.mean(low_freq_bins)
            features['high_freq_ratio'] = high_energy / low_energy if low_energy > 0 else 0
            
            # Spectral flux - rate of change of spectrum
            spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
            features['spectral_flux_std'] = np.std(spectral_flux)
            features['spectral_flux_mean'] = np.mean(spectral_flux)
            
        except Exception as e:
            # If analysis fails, return values that lean toward AI detection
            features = {
                'pitch_cv': 0.05,
                'pitch_range': 0.1,
                'jitter': 0.001,
                'spectral_centroid_std': 200,
                'spectral_centroid_mean': 2000,
                'spectral_centroid_cv': 0.1,
                'spectral_bandwidth_std': 100,
                'spectral_bandwidth_mean': 1500,
                'spectral_contrast_mean': 20,
                'spectral_contrast_std': 5,
                'spectral_rolloff_std': 200,
                'spectral_rolloff_mean': 3000,
                'spectral_flatness_mean': 0.1,
                'spectral_flatness_std': 0.02,
                'zcr_std': 0.02,
                'zcr_mean': 0.1,
                'rms_std': 0.02,
                'rms_mean': 0.1,
                'rms_cv': 0.2,
                'mfcc_var': 20,
                'mfcc_delta_var': 5,
                'mfcc_delta2_var': 2,
                'harmonic_ratio': 0.9,
                'chroma_std': 0.1,
                'silence_ratio': 0.05,
                'high_freq_ratio': 0.01,
                'spectral_flux_std': 10,
                'spectral_flux_mean': 50
            }
        
        return features
    
    @staticmethod
    def compute_ai_score(features: Dict[str, float]) -> float:
        """
        AGGRESSIVE AI detection scoring for modern AI voices.
        Modern AI voices (ElevenLabs, XTTS, Bark, etc.) are very good,
        so we use much more sensitive thresholds.
        """
        from loguru import logger
        
        ai_score = 0.0
        checks_triggered = []
        
        # Log all features for debugging
        logger.debug(f"Feature values: {features}")
        
        # ===== PITCH UNIFORMITY (Modern AI has good but still detectable uniformity) =====
        pitch_cv = features.get('pitch_cv', 0.15)
        logger.debug(f"pitch_cv: {pitch_cv}")
        if pitch_cv < 0.25:  # Much higher threshold - modern AI has up to 0.2 CV
            score = 1.0 - (pitch_cv / 0.25)  # Linear scale
            ai_score += score * 0.15
            if pitch_cv < 0.18:
                checks_triggered.append(f"pitch_cv={pitch_cv:.3f}")
        
        # Pitch range - AI typically has narrower range
        pitch_range = features.get('pitch_range', 0.4)
        logger.debug(f"pitch_range: {pitch_range}")
        if pitch_range < 0.5:  # Humans typically have wider range
            score = 1.0 - (pitch_range / 0.5)
            ai_score += score * 0.12
            if pitch_range < 0.35:
                checks_triggered.append(f"pitch_range={pitch_range:.3f}")
        
        # Jitter - micro-perturbations (humans have more)
        jitter = features.get('jitter', 0.015)
        logger.debug(f"jitter: {jitter}")
        if jitter < 0.02:  # Higher threshold
            score = 1.0 - (jitter / 0.02)
            ai_score += score * 0.12
            if jitter < 0.012:
                checks_triggered.append(f"jitter={jitter:.4f}")
        
        # ===== SPECTRAL CONSISTENCY =====
        spectral_cv = features.get('spectral_centroid_cv', 0.25)
        logger.debug(f"spectral_cv: {spectral_cv}")
        if spectral_cv < 0.30:
            score = 1.0 - (spectral_cv / 0.30)
            ai_score += score * 0.10
            if spectral_cv < 0.22:
                checks_triggered.append(f"spectral_cv={spectral_cv:.3f}")
        
        # Spectral contrast variation
        contrast_std = features.get('spectral_contrast_std', 12)
        logger.debug(f"contrast_std: {contrast_std}")
        if contrast_std < 15:
            score = 1.0 - (contrast_std / 15)
            ai_score += score * 0.08
            if contrast_std < 10:
                checks_triggered.append(f"contrast_std={contrast_std:.2f}")
        
        # ===== ENERGY DYNAMICS =====
        rms_cv = features.get('rms_cv', 0.5)
        logger.debug(f"rms_cv: {rms_cv}")
        if rms_cv < 0.6:  # Humans have more dynamic range
            score = 1.0 - (rms_cv / 0.6)
            ai_score += score * 0.10
            if rms_cv < 0.4:
                checks_triggered.append(f"rms_cv={rms_cv:.3f}")
        
        # Zero crossing rate variation
        zcr_std = features.get('zcr_std', 0.06)
        logger.debug(f"zcr_std: {zcr_std}")
        if zcr_std < 0.08:
            score = 1.0 - (zcr_std / 0.08)
            ai_score += score * 0.06
            if zcr_std < 0.04:
                checks_triggered.append(f"zcr_std={zcr_std:.4f}")
        
        # ===== MFCC DYNAMICS (Key for detecting AI) =====
        mfcc_var = features.get('mfcc_var', 60)
        logger.debug(f"mfcc_var: {mfcc_var}")
        if mfcc_var < 80:  # Higher threshold
            score = 1.0 - (mfcc_var / 80)
            ai_score += score * 0.10
            if mfcc_var < 50:
                checks_triggered.append(f"mfcc_var={mfcc_var:.2f}")
        
        # MFCC delta variance - transitions
        mfcc_delta_var = features.get('mfcc_delta_var', 15)
        logger.debug(f"mfcc_delta_var: {mfcc_delta_var}")
        if mfcc_delta_var < 20:
            score = 1.0 - (mfcc_delta_var / 20)
            ai_score += score * 0.12
            if mfcc_delta_var < 12:
                checks_triggered.append(f"mfcc_delta={mfcc_delta_var:.2f}")
        
        # MFCC delta2 variance
        mfcc_delta2_var = features.get('mfcc_delta2_var', 8)
        logger.debug(f"mfcc_delta2_var: {mfcc_delta2_var}")
        if mfcc_delta2_var < 10:
            score = 1.0 - (mfcc_delta2_var / 10)
            ai_score += score * 0.08
            if mfcc_delta2_var < 5:
                checks_triggered.append(f"mfcc_d2={mfcc_delta2_var:.2f}")
        
        # ===== HARMONIC STRUCTURE =====
        harmonic_ratio = features.get('harmonic_ratio', 0.75)
        logger.debug(f"harmonic_ratio: {harmonic_ratio}")
        if harmonic_ratio > 0.80:  # AI often too clean/harmonic
            score = (harmonic_ratio - 0.80) / 0.20
            ai_score += score * 0.08
            if harmonic_ratio > 0.88:
                checks_triggered.append(f"harmonic={harmonic_ratio:.3f}")
        
        # Chroma variation
        chroma_std = features.get('chroma_std', 0.18)
        logger.debug(f"chroma_std: {chroma_std}")
        if chroma_std < 0.22:
            score = 1.0 - (chroma_std / 0.22)
            ai_score += score * 0.06
            if chroma_std < 0.15:
                checks_triggered.append(f"chroma_std={chroma_std:.3f}")
        
        # ===== SPECTRAL CHARACTERISTICS =====
        sf = features.get('spectral_flatness_mean', 0.08)
        sf_std = features.get('spectral_flatness_std', 0.04)
        logger.debug(f"spectral_flatness: mean={sf}, std={sf_std}")
        
        # AI often has unusual flatness
        if sf < 0.05 or sf > 0.35:
            ai_score += 0.06
            checks_triggered.append(f"flatness={sf:.4f}")
        
        # Low flatness variation
        if sf_std < 0.05:
            score = 1.0 - (sf_std / 0.05)
            ai_score += score * 0.05
            if sf_std < 0.03:
                checks_triggered.append(f"flatness_std={sf_std:.4f}")
        
        # Spectral flux variation
        flux_std = features.get('spectral_flux_std', 25)
        logger.debug(f"flux_std: {flux_std}")
        if flux_std < 30:
            score = 1.0 - (flux_std / 30)
            ai_score += score * 0.06
            if flux_std < 18:
                checks_triggered.append(f"flux_std={flux_std:.2f}")
        
        # Bandwidth variation
        bw_std = features.get('spectral_bandwidth_std', 350)
        logger.debug(f"bandwidth_std: {bw_std}")
        if bw_std < 400:
            score = 1.0 - (bw_std / 400)
            ai_score += score * 0.06
            if bw_std < 250:
                checks_triggered.append(f"bw_std={bw_std:.1f}")
        
        # ===== HIGH FREQUENCY ANALYSIS =====
        high_freq_ratio = features.get('high_freq_ratio', 0.02)
        logger.debug(f"high_freq_ratio: {high_freq_ratio}")
        # AI may have unusual high-freq patterns
        if high_freq_ratio < 0.01 or high_freq_ratio > 0.08:
            ai_score += 0.05
            checks_triggered.append(f"hf_ratio={high_freq_ratio:.4f}")
        
        # ===== SILENCE PATTERNS =====
        silence_ratio = features.get('silence_ratio', 0.1)
        logger.debug(f"silence_ratio: {silence_ratio}")
        # AI often has more consistent silence patterns
        if silence_ratio < 0.05:  # Too little silence = unnatural
            ai_score += 0.04
            checks_triggered.append(f"silence={silence_ratio:.3f}")
        
        # Log scoring info
        logger.info(f"AI checks triggered ({len(checks_triggered)}): {checks_triggered}")
        logger.info(f"Raw AI score before scaling: {ai_score:.4f}")
        
        # Scale to 0-1 range (max possible score ~1.19)
        normalized_score = min(1.0, ai_score / 0.80)
        
        # Apply boost for multiple triggers
        if len(checks_triggered) >= 5:
            normalized_score = min(1.0, normalized_score * 1.3)
        elif len(checks_triggered) >= 3:
            normalized_score = min(1.0, normalized_score * 1.15)
        
        logger.info(f"Final spectral AI score: {normalized_score:.4f}")
        return normalized_score