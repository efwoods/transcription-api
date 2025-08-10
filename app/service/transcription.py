# app/service/transcription.py
import asyncio
import tempfile
import os
import gc
import numpy as np
import soundfile as sf
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from core.config import settings
from core.logging import logger
from models.whisper_model_base import get_model

# Limit number of concurrent transcriptions (reduces peak memory)
# Tune this to 1 or 2 depending on memory footprint and instance size.
_transcribe_semaphore = asyncio.Semaphore(1)

# Thread pool for blocking work (model.transcribe is blocking)
_executor = ThreadPoolExecutor(max_workers=2)


def _write_pcm_bytes_to_wav(tmp_path: str, pcm_bytes: bytes):
    """
    Expect 16-bit signed PCM mono, at settings.SAMPLE_RATE.
    Write WAV header and frames to tmp_path using wave module (low-overhead).
    """
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(settings.SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def _blocking_transcribe(tmp_path: str) -> Dict:
    """
    Blocking function to run model.transcribe synchronously.
    Intended to be executed in a thread via run_in_executor().
    """
    model_instance = get_model()
    # Default transcription args tuned for CPU/low memory:
    # - fp16 disabled (on CPU, do not use float16),
    # - temperature fixed,
    # - disable word timestamps if you don't need them (they increase memory).
    result = model_instance.transcribe(
        tmp_path,
        language="en",
        fp16=(settings.DEVICE == "cuda"),
        temperature=0.0,
        word_timestamps=False,  # set True only if you need timestamps
    )
    return result


async def transcribe_audio_chunk(pcm_bytes: bytes) -> Dict:
    """
    Asynchronous wrapper to transcribe a single short chunk of PCM audio bytes.
    This function is memory-conscious: writes to a small temporary file, executes
    transcription in a thread, then aggressively cleans up memory.
    """
    # Acquire semaphore to limit concurrent transcriptions
    async with _transcribe_semaphore:
        tmpfile = None
        try:
            # Create a NamedTemporaryFile on disk and close right away so whisper can open it.
            # delete=False so we can control removal (ensures cross-platform behavior).
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmpfile = tf.name
            # Write bytes to wav with header
            _write_pcm_bytes_to_wav(tmpfile, pcm_bytes)

            # Get simple pre-transcription diagnostics (use soundfile, low-overhead)
            try:
                data, sr = sf.read(tmpfile, dtype="float32")
                duration = float(len(data) / sr)
                amplitude = float(np.max(np.abs(data))) if data.size > 0 else 0.0
            except Exception:
                duration = None
                amplitude = None

            logger.info("Transcribing %s (duration=%s)", tmpfile, duration)

            # Call model.transcribe in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _executor, _blocking_transcribe, tmpfile
            )

            # Extract results safely
            transcript = (
                result.get("text", "").strip() if isinstance(result, dict) else ""
            )
            segments = result.get("segments", []) if isinstance(result, dict) else []
            no_speech_prob = segments[0].get("no_speech_prob", 0.0) if segments else 0.0
            avg_logprob = segments[0].get("avg_logprob", 0.0) if segments else 0.0

            return {
                "transcript": transcript,
                "duration": duration,
                "amplitude": amplitude,
                "no_speech_prob": float(no_speech_prob),
                "avg_logprob": float(avg_logprob),
            }

        except Exception as exc:
            logger.error("Error in transcribe_audio_chunk: %s", exc)
            return {"error": str(exc)}
        finally:
            # Remove temp file and force GC
            if tmpfile and os.path.exists(tmpfile):
                try:
                    os.remove(tmpfile)
                except Exception:
                    logger.warning("Failed to remove tmpfile %s", tmpfile)
            # Delete references and run GC to free memory
            try:
                del pcm_bytes
            except Exception:
                pass
            gc.collect()
