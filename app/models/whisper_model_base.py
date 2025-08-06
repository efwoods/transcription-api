import whisper
from app.core.config import settings
from app.core.logging import logger

# Initialize Whisper model
_model = None  # private singleton instance


def init():
    global _model
    if _model is None:
        _model = whisper.load_model("tiny", device=settings.DEVICE)
        logger.info("Whisper model initialized")
    else:
        logger.info("Whisper model already initialized")


def model():
    if _model is None:
        init()
    return _model
