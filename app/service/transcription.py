import numpy as np
import tempfile
import wave
import soundfile as sf
from app.core.config import settings
from app.core.logging import logger
from app.models.whisper_model_base import model


async def transcribe_audio(audio_data: bytes) -> dict:
    model_instance = model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(settings.SAMPLE_RATE)
            wf.writeframes(audio_data)

        try:
            data, samplerate = sf.read(tmpfile.name)
            duration = len(data) / samplerate
            amplitude = float(np.max(np.abs(data)))
            logger.info(
                f"Audio duration: {duration:.2f} sec, Max amplitude: {amplitude}"
            )
            # Normalize the audio data
            data = data / np.max(np.abs(data))
            # Update the data with a normalized audio file.
            sf.write(tmpfile.name, data, samplerate, subtype="PCM_16")
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return {"error": str(e)}

        logger.info(f"Transcribing {tmpfile.name}")
        result = model_instance.transcribe(
            tmpfile.name,
            language="en",
            fp16=settings.DEVICE == "cuda",
            temperature=0.0,
            word_timestamps=True,
        )
        transcript = result.get("text", "").strip()
        segments = result.get("segments", [])
        confidence = segments[0].get("no_speech_prob", 0.0) if segments else 0.0
        avg_logprob = segments[0].get("avg_logprob", 0.0) if segments else 0.0
        logger.info(
            f"Transcript: {transcript}, Confidence: {confidence}, Logprob: {avg_logprob}"
        )

        return {
            "transcript": transcript,
            "duration": duration,
            "amplitude": amplitude,
            "no_speech_prob": confidence,
            "avg_logprob": avg_logprob,
        }
