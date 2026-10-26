import librosa
import numpy as np
import tensorflow as tf
from keras import layers

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 40
WINDOW_SAMPLES = SR
WINDOW_FRAMES = WINDOW_SAMPLES // HOP + 1


def mel_power(audio: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )


def to_logmel(power: np.ndarray) -> np.ndarray:
    db = librosa.power_to_db(power, ref=np.max(power) + 1e-10, top_db=80.0)
    return ((db + 80.0) / 80.0).T.astype(np.float32)


def clip_features(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return to_logmel(mel_power(audio[:WINDOW_SAMPLES]))


def build_filler_model(input_shape: tuple = (WINDOW_FRAMES, N_MELS, 1)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="logmel_input")
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        # GroupNormalization, not BatchNormalization: BatchNorm's running statistics
        # drifted badly on this data, collapsing validation accuracy to ~50%.
        x = layers.GroupNormalization(groups=8)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="filler_prob")(x)
    return tf.keras.Model(inputs, outputs, name="filler_detector")


STEP_FRAMES = 10


def window_probabilities(audio: np.ndarray, model: tf.keras.Model) -> tuple:
    """Filler probability for a 1 s window every 0.1 s, plus each window's centre time
    and whether its centre is voiced (not silence)."""
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))

    power = mel_power(audio)
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP)[0]
    silence_level = 0.05 * (np.max(rms) + 1e-10)

    starts = list(range(0, power.shape[1] - WINDOW_FRAMES + 1, STEP_FRAMES))
    windows = np.stack([to_logmel(power[:, s:s + WINDOW_FRAMES]) for s in starts])
    probs = model.predict(windows[..., np.newaxis], batch_size=256, verbose=0)[:, 0]

    # The clips the model learned from have the filler in the middle of the window.
    centre = WINDOW_FRAMES // 2
    centres = np.array([(s + centre) * HOP / SR for s in starts])
    voiced = np.array([rms[s + centre - 5:s + centre + 5].max() >= silence_level for s in starts])
    return probs, centres, voiced


def find_events(probs: np.ndarray, centres: np.ndarray, voiced: np.ndarray,
                threshold: float = 0.5, min_windows: int = 2) -> list:
    """Consecutive windows above the threshold belong to the same "uh"/"um", so each
    run of at least `min_windows` positive windows is counted once."""
    positive = (probs >= threshold) & voiced
    times = []
    run = []
    for i, is_pos in enumerate(list(positive) + [False]):
        if is_pos:
            run.append(i)
            continue
        if len(run) >= min_windows:
            peak = run[int(np.argmax(probs[run]))]
            times.append(round(float(centres[peak]), 2))
        run = []
    return times


def detect_fillers(audio: np.ndarray, model: tf.keras.Model, threshold: float = 0.5,
                   min_windows: int = 2) -> dict:
    probs, centres, voiced = window_probabilities(audio, model)
    times = find_events(probs, centres, voiced, threshold, min_windows)
    return {"filler_count": len(times), "filler_times_sec": times}
