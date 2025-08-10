import threading
import gc
import whisper
from core.config import settings
from core.logging import logger

# Private module-level singletons
_model = None
_model_lock = threading.Lock()


def init(model_name: str = "tiny"):
    """
    Initialize the Whisper model singleton. Thread-safe.
    Call this once at startup (lifespan), not per request.
    """
    global _model
    if _model is not None:
        logger.info("Whisper model already initialized")
        return _model

    with _model_lock:
        if _model is None:
            logger.info(
                "Loading Whisper model '%s' on device=%s", model_name, settings.DEVICE
            )
            # On CPU: fp16 is not used. On CUDA you may enable fp16 in transcription call.
            _model = whisper.load_model(model_name, device=settings.DEVICE)
            logger.info("Whisper model initialized")
    return _model


def get_model():
    """
    Return the model, initializing lazily if needed.
    Prefer calling init() in the application lifespan startup.
    """
    global _model
    if _model is None:
        return init()
    return _model


def shutdown():
    """
    Optional explicit shutdown hook; release model and force GC.
    """
    global _model
    with _model_lock:
        if _model is not None:
            logger.info("Releasing Whisper model from memory")
            _model = None
            gc.collect()
