"""
ML Models for Clarivo
Transcription, scoring, and feedback generation
"""
import random
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

from config import config
from utils import AudioProcessor, TextNormalizer, split_into_syllables

# Import ML libraries
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class TranscriptionResult:
    text: str
    confidence: float = 0.0
    language: str = "en"


class WhisperTranscriber:
    def __init__(self, model_size: str = None):
        self.model_size = model_size or config.whisper_model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not installed. Run: pip install openai-whisper")
        print(f"📥 Loading Whisper {self.model_size} model...")
        self.model = whisper.load_model(self.model_size, device=self.device)
        print(f"✅ Whisper loaded on {self.device}")
    
    def transcribe(self, audio_path: str, target_phrase: str = None) -> TranscriptionResult:
        if self.model is None:
            self.load_model()
        
        # Load audio using librosa (avoid ffmpeg dependency)
        try:
            import librosa
            audio_data = librosa.load(audio_path, sr=16000)[0]
        except:
            # Fallback to direct path if librosa fails
            audio_data = audio_path
        
        options = {"language": "en", "task": "transcribe"}
        if target_phrase:
            options["initial_prompt"] = f"The speaker is trying to say: {target_phrase}"
        
        result = self.model.transcribe(audio_data, **options)
        
        # Use a conservative confidence (not relied upon heavily)
        confidence = 0.7  # Default baseline
        if "segments" in result and result["segments"]:
            no_speech_probs = [seg.get("no_speech_prob", 0.0) for seg in result["segments"]]
            avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
            confidence = max(0.3, 1.0 - avg_no_speech)  # Floor at 0.3
        
        return TranscriptionResult(
            text=result["text"].strip(),
            confidence=confidence,
            language=result.get("language", "en")
        )


class FeedbackColor(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class ScoringResult:
    final_score: int
    color: FeedbackColor
    scores: Dict[str, float]
    target_text: str = ""
    transcribed_text: str = ""
    word_comparison: list = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_score": self.final_score,
            "color": self.color.value,
            "scores": self.scores,
            "target": self.target_text,
            "transcribed": self.transcribed_text,
            "word_comparison": self.word_comparison or []
        }


class CombinedScorer:
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.audio_processor = AudioProcessor()
    
    def score(self, target: str, transcription: str, confidence: float, 
              audio_path: str = None, streak: int = 0) -> ScoringResult:
        
        target_norm = self.normalizer.normalize(target, for_aphasia=True)
        trans_norm = self.normalizer.normalize(transcription, for_aphasia=True)
        
        # WER/CER
        wer_score = self._calculate_wer(target_norm, trans_norm)
        cer_score = self._calculate_cer(target_norm, trans_norm)
        
        # Calculate WPM
        wpm = 0
        audio_duration = 0
        
        # Prosody (if audio available)
        prosody_score = 0.6  # Default
        if audio_path and SOUNDFILE_AVAILABLE:
            try:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Calculate audio duration and WPM
                audio_duration = len(audio) / sr
                word_count = len(trans_norm.split())
                if audio_duration > 0:
                    wpm = int((word_count / audio_duration) * 60)
                
                features = self.audio_processor.extract_prosody_features(audio, sr)
                # Weighted prosody: reward natural pace, penalize excessive pauses
                prosody_score = 0.7 * features["speech_rate"] + 0.3 * (1 - min(features["pause_ratio"], 0.5))
            except:
                pass
        
        # Word-level comparison
        word_comparison = self._compare_words(target_norm, trans_norm)
        
        # Adaptive threshold based on streak (easier for beginners)
        adaptive_green = max(config.min_green_threshold, 
                            config.base_green_threshold - 0.02 * max(0, 5 - streak))
        
        # Final score (NO semantic, NO confidence)
        final = (
            wer_score * config.wer_weight +
            cer_score * config.cer_weight +
            prosody_score * config.prosody_weight
        ) * 100
        final_score = int(round(final))
        
        # Determine color with adaptive threshold
        if final_score >= adaptive_green * 100:
            color = FeedbackColor.GREEN
        elif final_score >= config.yellow_threshold * 100:
            color = FeedbackColor.YELLOW
        else:
            color = FeedbackColor.RED
        
        return ScoringResult(
            final_score=final_score,
            color=color,
            scores={
                "word_accuracy": wer_score * 100,
                "character_accuracy": cer_score * 100,
                "prosody": prosody_score * 100,
                "adaptive_threshold": adaptive_green * 100,
                "wpm": wpm,
                "duration": round(audio_duration, 2)
            },
            target_text=target_norm,
            transcribed_text=trans_norm,
            word_comparison=word_comparison
        )
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        if not JIWER_AVAILABLE or not reference or not hypothesis:
            return 0.5
        try:
            wer = jiwer.wer(reference, hypothesis)
            return max(0.0, 1.0 - wer)
        except:
            return 0.5
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        if not JIWER_AVAILABLE or not reference or not hypothesis:
            return 0.5
        try:
            cer = jiwer.cer(reference, hypothesis)
            return max(0.0, 1.0 - cer)
        except:
            return 0.5
    
    def _compare_words(self, target: str, transcribed: str) -> list:
        """Compare words and identify errors with syllable breakdown"""
        target_words = target.split()
        trans_words = transcribed.split()
        comparison = []
        
        # Simple word-by-word alignment
        max_len = max(len(target_words), len(trans_words))
        
        for i in range(max_len):
            expected = target_words[i] if i < len(target_words) else ''
            spoken = trans_words[i] if i < len(trans_words) else ''
            
            is_correct = expected.lower() == spoken.lower()
            
            comparison.append({
                'expected': expected,
                'spoken': spoken,
                'correct': is_correct,
                'expected_syllables': split_into_syllables(expected) if expected else [],
                'spoken_syllables': split_into_syllables(spoken) if spoken else []
            })
        
        return comparison


class FeedbackGenerator:
    def generate(self, score: int, color: FeedbackColor, streak: int = 0) -> str:
        if color == FeedbackColor.GREEN:
            message = random.choice(config.success_messages)
            if streak >= 3:
                message += f" 🔥 {streak} in a row!"
        elif color == FeedbackColor.YELLOW:
            message = random.choice(config.partial_messages)
        else:
            message = random.choice(config.retry_messages)
        return message


class TTSEngine:
    def __init__(self):
        self.slow = config.tts_slow
        self.lang = config.tts_lang
    
    def generate_audio(self, text: str) -> str:
        if not GTTS_AVAILABLE:
            raise ImportError("gTTS not installed. Run: pip install gtts")
        tts = gTTS(text=text, lang=self.lang, slow=self.slow)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tts.save(f.name)
            return f.name