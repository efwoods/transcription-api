from fastapi import APIRouter, WebSocket
from service.transcription import transcribe_audio
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
import json
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, RedirectResponse

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

@router.get("/ws-info", tags=["Transcribe"])
async def websocket_info():
    return {
        "endpoint": "/ws",
        "protocol": "WebSocket",
        "description": "Real-time audio transcription WebSocket endpoint.",
        "input": "Raw PCM audio bytes streamed in small chunks.",
        "output": "JSON messages containing transcription text.",
        "note": "Each message must be mono 16-bit PCM, chunked according to SAMPLE_RATE * CHUNK_DURATION * 2.",
    }
