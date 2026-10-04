"""
preprocessing.py
-----------------
Audio preprocessing for the Presentation Coaching System.

Pipeline (matches Section 3.5.3 of the proposal):
    1. Load audio + resample to 16kHz (standard rate for speech models)
    2. Reduce background noise (noisereduce)
    3. Trim leading/trailing silence
    4. Normalize amplitude

Usage:
    from preprocessing import preprocess_audio
    audio, sr = preprocess_audio("presentation.wav")
"""

import librosa
import noisereduce as nr
import numpy as np

TARGET_SR = 16000


def load_audio(path: str, target_sr: int = TARGET_SR):
    """Load an audio file and resample it to target_sr."""
    audio, sr = librosa.load(path, sr=target_sr)
    return audio, sr


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply spectral-gating noise reduction."""
    return nr.reduce_noise(y=audio, sr=sr)


def trim_silence(audio: np.ndarray, top_db: int = 25) -> np.ndarray:
    """Remove leading/trailing silence. Does NOT remove mid-speech pauses,
    since pause duration is itself a feature we want to keep (see features.py)."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    return librosa.util.normalize(audio)


def preprocess_audio(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline for one audio file."""
    audio, sr = load_audio(path, target_sr)
    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio)
    audio = normalize(audio)
    return audio, sr


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_audio.wav>")
        sys.exit(1)

    audio, sr = preprocess_audio(sys.argv[1])
    print(f"Processed audio: {len(audio) / sr:.2f} seconds at {sr} Hz")
