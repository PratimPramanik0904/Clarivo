"""
Utility functions for Clarivo
Audio processing and text normalization
"""
import io
import re
import tempfile
from pathlib import Path
from typing import Tuple, Dict
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from config import config


class AudioProcessor:
    """Handles audio loading and preprocessing"""
    
    def __init__(self):
        self.sample_rate = config.sample_rate
    
    def load_audio_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        if not SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile required. Install: pip install soundfile")
        
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)
        return audio.astype(np.float32), self.sample_rate
    
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        threshold = config.silence_threshold
        non_silent = np.abs(audio) > threshold
        if np.any(non_silent):
            start = np.argmax(non_silent)
            end = len(audio) - np.argmax(non_silent[::-1])
            audio = audio[start:end]
        return audio
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        if LIBROSA_AVAILABLE:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        return np.interp(
            np.linspace(0, len(audio), target_length),
            np.arange(len(audio)),
            audio
        )
    
    def save_temp_wav(self, audio: np.ndarray, sample_rate: int = None) -> str:
        sr = sample_rate or self.sample_rate
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            return f.name
    
    def extract_prosody_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract speech rate and pause ratio for aphasia assessment"""
        if not LIBROSA_AVAILABLE or len(audio) == 0:
            return {"speech_rate": 0.6, "pause_ratio": 0.2}
        
        duration = len(audio) / sr
        if duration == 0:
            return {"speech_rate": 0.6, "pause_ratio": 0.2}
        
        # Estimate word count (conservative: 1 word per 0.6s)
        estimated_words = max(1, duration / 0.6)
        # Normalize speech rate to [0,1] (1.0 = natural pace)
        speech_rate = min(1.0, estimated_words / (duration / 0.5))
        
        # Pause detection via energy
        energy = librosa.feature.rms(y=audio)[0]
        silent_frames = energy < config.silence_threshold
        pause_ratio = float(np.mean(silent_frames)) if len(silent_frames) > 0 else 0.2
        
        return {
            "speech_rate": float(speech_rate),
            "pause_ratio": float(pause_ratio)
        }


def split_into_syllables(word: str) -> list:
    """Simple syllable splitting (vowel-based heuristic)"""
    if not word:
        return []
    
    vowels = 'aeiouAEIOU'
    syllables = []
    current = ''
    
    for i, char in enumerate(word):
        current += char
        # Split after vowel if next char is consonant
        if char in vowels and i < len(word) - 1 and word[i + 1] not in vowels:
            syllables.append(current)
            current = ''
    
    if current:
        syllables.append(current)
    
    return syllables if syllables else [word]


class TextNormalizer:
    """Normalize text for comparison"""
    
    APHASIA_EQUIVALENTS = {
        "gonna": "going to", "wanna": "want to", "gotta": "got to",
        "kinda": "kind of", "sorta": "sort of", "dunno": "don't know",
        "gimme": "give me", "lemme": "let me", "cause": "because",
        "'cause": "because", "yeah": "yes", "yep": "yes",
        "nope": "no", "nah": "no", "uh": "", "um": "", "ah": "",
        "er": "", "like": "", "okay": "ok"
    }
    
    def normalize(self, text: str, for_aphasia: bool = True) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        if for_aphasia:
            for informal, formal in self.APHASIA_EQUIVALENTS.items():
                text = re.sub(r'\b' + re.escape(informal) + r'\b', formal, text)
        text = ' '.join(text.split())
        return text.strip()