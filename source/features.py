import librosa
import numpy as np

FILLER_WORDS = {"um", "uh", "umm", "uhh", "like", "ah", "aaa", "err"}


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T


def extract_pitch(audio: np.ndarray, sr: int) -> dict:
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=65.0, fmax=400.0, sr=sr
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return {"mean_pitch_hz": 0.0, "pitch_std_hz": 0.0}
    return {
        "mean_pitch_hz": float(np.mean(voiced_f0)),
        "pitch_std_hz": float(np.std(voiced_f0)),
    }


def detect_pauses(audio: np.ndarray, sr: int, top_db: int = 30, min_pause_sec: float = 0.3) -> dict:
    intervals = librosa.effects.split(audio, top_db=top_db)
    pauses = []
    for i in range(1, len(intervals)):
        gap_start = intervals[i - 1][1]
        gap_end = intervals[i][0]
        gap_sec = (gap_end - gap_start) / sr
        if gap_sec >= min_pause_sec:
            pauses.append(gap_sec)

    return {
        "pause_count": len(pauses),
        "total_pause_time_sec": float(sum(pauses)),
        "avg_pause_sec": float(np.mean(pauses)) if pauses else 0.0,
    }


def estimate_speech_rate(word_count: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return (word_count / duration_sec) * 60.0


def detect_filler_words(transcript_words: list[str]) -> dict:
    words_lower = [w.lower().strip(".,!?") for w in transcript_words]
    fillers_found = [w for w in words_lower if w in FILLER_WORDS]
    return {
        "filler_count": len(fillers_found),
        "filler_words": fillers_found,
    }


def extract_all_features(audio: np.ndarray, sr: int, transcript_words: list[str] | None = None) -> dict:
    duration_sec = len(audio) / sr

    result = {
        "duration_sec": duration_sec,
        "pitch": extract_pitch(audio, sr),
        "pauses": detect_pauses(audio, sr),
        "mfcc_shape": extract_mfcc(audio, sr).shape,
    }

    if transcript_words is not None:
        result["speech_rate_wpm"] = estimate_speech_rate(len(transcript_words), duration_sec)
        result["fillers"] = detect_filler_words(transcript_words)
    else:
        result["speech_rate_wpm"] = None
        result["fillers"] = None
        result["_note"] = "Pass transcript_words (from ASR) to get speech rate + filler detection."

    return result


if __name__ == "__main__":
    import sys
    from preprocessing import preprocess_audio

    if len(sys.argv) < 2:
        print("Usage: python features.py <path_to_audio.wav>")
        sys.exit(1)

    audio, sr = preprocess_audio(sys.argv[1])
    features = extract_all_features(audio, sr)
    for k, v in features.items():
        print(f"{k}: {v}")
