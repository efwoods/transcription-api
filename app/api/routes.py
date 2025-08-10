# app/api/routes.py
from fastapi import APIRouter, WebSocket
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
import json
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, RedirectResponse
from service.transcription import transcribe_audio_chunk

router = APIRouter()

# chunk_size in bytes for 16-bit mono audio
CHUNK_SIZE_BYTES = int(settings.SAMPLE_RATE * settings.CHUNK_DURATION * 2)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    metrics.active_websockets.inc()
    try:
        buffer = bytearray()
        processing = False
        while True:
            # await next binary payload from client
            data = await websocket.receive_bytes()
            if not data:
                continue
            buffer.extend(data)

            # Process as many full chunks as present
            while len(buffer) >= CHUNK_SIZE_BYTES:
                chunk = bytes(buffer[:CHUNK_SIZE_BYTES])
                # remove chunk from buffer
                del buffer[:CHUNK_SIZE_BYTES]

                # Transcribe chunk
                # The transcription function internally uses a semaphore to rate-limit concurrency.
                result = await transcribe_audio_chunk(chunk)

                # Send result back
                await websocket.send_text(json.dumps(result))
                metrics.transcriptions_processed.inc()

            # If buffer grows too large (client sending faster than processing),
            # optionally drop oldest data to avoid OOM. Here we cap buffer at 4 chunks.
            max_buffer_bytes = CHUNK_SIZE_BYTES * 4
            if len(buffer) > max_buffer_bytes:
                # drop the oldest bytes (simple backpressure strategy)
                drop_bytes = len(buffer) - max_buffer_bytes
                del buffer[:drop_bytes]
                logger.warning(
                    "Websocket buffer exceeded %d bytes; dropped %d bytes",
                    max_buffer_bytes,
                    drop_bytes,
                )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        metrics.websocket_errors.inc()
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
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
