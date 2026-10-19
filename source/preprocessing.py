import librosa
import noisereduce as nr
import numpy as np

TARGET_SR = 16000


def load_audio(path: str, target_sr: int = TARGET_SR):
    audio, sr = librosa.load(path, sr=target_sr)
    return audio, sr


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    return nr.reduce_noise(y=audio, sr=sr)


def trim_silence(audio: np.ndarray, top_db: int = 25) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize(audio: np.ndarray) -> np.ndarray:
    return librosa.util.normalize(audio)


def preprocess_audio(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
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
