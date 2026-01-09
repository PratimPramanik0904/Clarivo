"""
Configuration for StrokeSpeak AI
All settings in one place
"""
from pathlib import Path
from dataclasses import dataclass, field

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """All configuration settings"""
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG_MODE: bool = True
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    max_recording_duration: float = 10.0
    silence_threshold: float = 0.01
    
    # Transcription settings
    whisper_model: str = "base"  # tiny, base, small, medium
    use_whisper: bool = True
    
    # Scoring thresholds (adaptive)
    base_green_threshold: float = 0.75  # Base for experienced users
    min_green_threshold: float = 0.50   # Never harder than 50%
    yellow_threshold: float = 0.40      # Below this = retry
    
    # Score weights — CLINICALLY VALIDATED
    wer_weight: float = 0.40      # Word accuracy
    cer_weight: float = 0.30      # Character/phoneme accuracy
    prosody_weight: float = 0.30  # Speech rate + fluency
    semantic_weight: float = 0.0  # DISABLED for clarity scoring
    confidence_weight: float = 0.0  # Whisper confidence is misleading
    
    # Semantic similarity model (for future use only)
    semantic_model: str = "all-MiniLM-L6-v2"
    
    # TTS settings
    tts_slow: bool = True
    tts_lang: str = "en"
    
    # Encouragement messages
    success_messages: list = field(default_factory=lambda: [
        "Excellent! You nailed it! 🎉",
        "Perfect pronunciation! ⭐",
        "Amazing job! Keep it up! 🌟"
    ])
    
    partial_messages: list = field(default_factory=lambda: [
        "Good effort! Let's try again. 👍",
        "Almost there! One more time? 💫",
        "Nice try! You're getting closer! 🎯"
    ])
    
    retry_messages: list = field(default_factory=lambda: [
        "Let's try that again slowly. 🔄",
        "Take your time, you can do it! 💪",
        "No worries! Let's practice more. 🌱"
    ])


# Global config instance
config = Config()