"""
StrokeSpeak AI - Main Entry Point
Speech therapy application for aphasia patients
Run with: python main.py
"""
import uvicorn
from config import config

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG_MODE
    )
