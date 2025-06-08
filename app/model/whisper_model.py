import whisper
from core.config import settings
from core.logging import logger

# Initialize Whisper model
_model = None  # private singleton instance

def init():
    global _model
    if _model is None:
        _model = whisper.load_model("base", device=settings.DEVICE)  
        logger.info("Whisper model initialized")
    else:
        logger.info("Whisper model already initialized")

def model():
    if _model is None:
        init()
    return _model