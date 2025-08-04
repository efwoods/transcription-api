# preprocess_audio.py
from pydub import AudioSegment
import os

def preprocess_audio(input_path, output_path):
    """
    Preprocess audio to improve quality for Whisper transcription.
    Args:
        input_path (str): Path to input audio file.
        output_path (str): Path to save preprocessed audio.
    """
    # Load audio
    audio = AudioSegment.from_file(input_path)

    # Normalize audio to -20 dBFS
    audio = audio.normalize()

    # Apply high-pass and low-pass filters to reduce noise
    audio = audio.high_pass_filter(100).low_pass_filter(3000)

    # Trim silence (threshold: -40 dBFS, min silence: 500ms)
    audio = audio.strip_silence(silence_len=500, silence_thresh=-40, padding=100)

    # Resample to 16kHz
    audio = audio.set_frame_rate(16000)

    # Export preprocessed audio
    audio.export(output_path, format="wav")
    print(f"Preprocessed audio saved to {output_path}")

if __name__ == "__main__":
    input_audio = "input.wav"  # Replace with your audio file
    output_audio = "preprocessed.wav"
    preprocess_audio(input_audio, output_audio)