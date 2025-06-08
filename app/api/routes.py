from fastapi import APIRouter
from fastapi import WebSocket
from service.transcription import transcribe_audio
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
import json

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    metrics.active_websockets.inc()
    try:
        buffer = bytearray()
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)
            chunk_size = settings.SAMPLE_RATE * settings.CHUNK_DURATION * 2  # 16-bit mono
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                result = await transcribe_audio(chunk)
                await websocket.send_text(json.dumps(result))
                metrics.transcriptions_processed.inc()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        metrics.websocket_errors.inc()
    finally:
        await websocket.close()
        metrics.active_websockets.dec()