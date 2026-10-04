"""
features.py
------------
Extracts the speaking-performance features your proposal evaluates:
    - Speech rate (words/syllables per minute)
    - Pitch variation
    - Pauses (count + duration)
    - Filler words (stub - needs ASR transcript, see note below)
    - MFCC features (fed into the Conformer model itself)

These features feed two separate things:
    1. The dashboard metrics shown on the Real-Time Analysis / Results pages
    2. Optionally, extra input signals alongside the raw audio for the
       Conformer model if you want a hybrid (acoustic + handcrafted) approach

Usage:
    from features import extract_all_features
    result = extract_all_features(audio, sr)
"""

import librosa
import numpy as np

FILLER_WORDS = {"um", "uh", "umm", "uhh", "like", "ah", "aaa", "err"}


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """MFCCs - the standard input representation fed into the Conformer."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape: (time_steps, n_mfcc)


def extract_pitch(audio: np.ndarray, sr: int) -> dict:
    """
    Pitch (F0) tracking using the pYIN algorithm.
    Returns mean pitch and pitch variation (std dev), which is what your
    'Pitch: Good / Natural Tone' UI metric is derived from.

    fmin/fmax are deliberately narrow (65-400 Hz) to match realistic human
    speaking voice range. A wider range (e.g. up to C7 ~2093 Hz) lets pyin
    occasionally lock onto noise or harmonics well outside real vocal
    range on lower-quality audio, and those rare outlier frames can blow
    up the standard deviation to physically implausible values (observed
    producing pitch_std_hz in the 600-900+ range on real data - not
    possible for an actual human voice).
    """
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
    """
    Detects speech segments vs. silence, then infers pauses as the gaps
    between segments that exceed min_pause_sec.
    """
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
    """
    Words-per-minute. Requires a word count, which normally comes from an
    ASR transcript (e.g. Whisper output) rather than raw audio alone.
    """
    if duration_sec <= 0:
        return 0.0
    return (word_count / duration_sec) * 60.0


def detect_filler_words(transcript_words: list[str]) -> dict:
    """
    NOTE: This is a simple keyword-matching stub, not the real filler-word
    model. Real filler-word detection (per your literature review - Yoodli,
    Speaker Coach, Orai all do this) needs either:
        (a) An ASR transcript + keyword spotting on it (fast to build, this
            function does that part), or
        (b) A dedicated acoustic classifier trained on PodcastFillers, which
            is more accurate on 'um/uh' that ASR sometimes drops or garbles.

    For your FYP timeline, (a) is the practical starting point. You can
    upgrade to (b) later if time permits, citing PodcastFillers as the
    training set.
    """
    words_lower = [w.lower().strip(".,!?") for w in transcript_words]
    fillers_found = [w for w in words_lower if w in FILLER_WORDS]
    return {
        "filler_count": len(fillers_found),
        "filler_words": fillers_found,
    }


def extract_all_features(audio: np.ndarray, sr: int, transcript_words: list[str] | None = None) -> dict:
    """Runs the full feature-extraction pass and returns one combined dict."""
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
