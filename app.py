"""
StrokeSpeak AI API
FastAPI application with all routes
"""
import base64
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import config, DATA_DIR, BASE_DIR
from utils import AudioProcessor
from models import WhisperTranscriber, CombinedScorer, FeedbackGenerator, TTSEngine


# ============= SCHEMAS =============

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]


class TranscribeRequest(BaseModel):
    audio_base64: str
    target_phrase: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    confidence: float


class PracticeRequest(BaseModel):
    audio_base64: str
    target_phrase: str
    use_whisper: bool = True
    generate_audio: bool = False
    streak: int = 0


class PracticeResponse(BaseModel):
    transcription: Dict[str, Any]
    scoring: Dict[str, Any]
    feedback: Dict[str, Any]


class ExerciseResponse(BaseModel):
    id: int
    phrase: str
    difficulty: str


# ============= FASTAPI APP =============

app = FastAPI(
    title="Clarivo",
    description="Speech therapy for aphasia patients",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
_whisper: Optional[WhisperTranscriber] = None
_scorer: Optional[CombinedScorer] = None
_feedback_gen: Optional[FeedbackGenerator] = None
_tts: Optional[TTSEngine] = None
_audio_proc: Optional[AudioProcessor] = None


def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperTranscriber()
        _whisper.load_model()
    return _whisper


def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = CombinedScorer()
    return _scorer


def get_feedback():
    global _feedback_gen
    if _feedback_gen is None:
        _feedback_gen = FeedbackGenerator()
    return _feedback_gen


def get_tts():
    global _tts
    if _tts is None:
        _tts = TTSEngine()
    return _tts


def get_audio_processor():
    global _audio_proc
    if _audio_proc is None:
        _audio_proc = AudioProcessor()
    return _audio_proc


# ============= ROUTES =============

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    frontend_path = BASE_DIR / "frontend" / "index.html"
    if not frontend_path.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(frontend_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "whisper": _whisper is not None,
            "scorer": _scorer is not None
        }
    )


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(request: TranscribeRequest):
    """Transcribe audio to text"""
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Load and preprocess audio
        processor = get_audio_processor()
        audio, sr = processor.load_audio_bytes(audio_bytes)
        audio = processor.preprocess(audio)
        
        # Save to temp file
        audio_path = processor.save_temp_wav(audio, sr)
        
        # Transcribe
        whisper = get_whisper()
        result = whisper.transcribe(audio_path, request.target_phrase)
        
        return TranscribeResponse(
            text=result.text,
            confidence=result.confidence
        )
        
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {str(e)}")


@app.post("/api/practice", response_model=PracticeResponse)
async def practice_pipeline(request: PracticeRequest):
    """Full pipeline: transcribe → score → feedback"""
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Process audio
        processor = get_audio_processor()
        audio, sr = processor.load_audio_bytes(audio_bytes)
        audio = processor.preprocess(audio)
        audio_path = processor.save_temp_wav(audio, sr)
        
        # Transcribe
        whisper = get_whisper()
        transcription = whisper.transcribe(audio_path, request.target_phrase)
        
        # Score (PASS audio_path for prosody analysis)
        scorer = get_scorer()
        scoring = scorer.score(
            request.target_phrase,
            transcription.text,
            transcription.confidence,
            audio_path=audio_path,  # Critical for prosody features
            streak=request.streak
        )
        
        # Generate feedback
        feedback_gen = get_feedback()
        feedback_message = feedback_gen.generate(
            scoring.final_score,
            scoring.color,
            request.streak
        )
        
        # Optional TTS
        audio_feedback_url = None
        if request.generate_audio:
            tts = get_tts()
            audio_path = tts.generate_audio(feedback_message)
            audio_feedback_url = f"/audio/{Path(audio_path).name}"
        
        return PracticeResponse(
            transcription={
                "text": transcription.text,
                "confidence": transcription.confidence
            },
            scoring=scoring.to_dict(),
            feedback={
                "message": feedback_message,
                "audio_url": audio_feedback_url
            }
        )
        
    except Exception as e:
        import traceback
        print(f"ERROR in /api/practice: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(500, f"Practice pipeline error: {str(e)}")


@app.get("/api/exercises", response_model=List[ExerciseResponse])
async def get_exercises():
    """Get practice sentences"""
    
    # Load from file if exists
    exercises_file = DATA_DIR / "exercises.json"
    if exercises_file.exists():
        with open(exercises_file) as f:
            data = json.load(f)
            return [ExerciseResponse(**ex) for ex in data]
    
    # Default exercises
    default_exercises = [
        {"id": 1, "phrase": "Hello, how are you today?", "difficulty": "easy"},
        {"id": 2, "phrase": "I want a glass of water please", "difficulty": "easy"},
        {"id": 3, "phrase": "Thank you very much for your help", "difficulty": "medium"},
        {"id": 4, "phrase": "What time is the meeting tomorrow?", "difficulty": "medium"},
        {"id": 5, "phrase": "The weather is beautiful today", "difficulty": "easy"},
        {"id": 6, "phrase": "Can you help me with this task?", "difficulty": "medium"},
        {"id": 7, "phrase": "I am feeling much better now", "difficulty": "easy"},
        {"id": 8, "phrase": "Please pass me the salt and pepper", "difficulty": "medium"},
        {"id": 9, "phrase": "The quick brown fox jumps over the lazy dog", "difficulty": "hard"},
        {"id": 10, "phrase": "I need to go to the doctor", "difficulty": "easy"},
    ]
    
    return [ExerciseResponse(**ex) for ex in default_exercises]


# ============= STARTUP =============

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    print("\n" + "="*60)
    print("    🗣️  Clarivo")
    print("    Speech Therapy for Aphasia Patients")
    print("="*60)
    print(f"  API Server: http://localhost:8000")
    print(f"  API Docs:   http://localhost:8000/docs")
    print(f"  Frontend:   http://localhost:8000")
    print("="*60 + "\n")
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)