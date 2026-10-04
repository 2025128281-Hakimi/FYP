"""
main.py
-------
Minimal Flask backend for the Presentation Coaching System.

This is the "Controller" in the MVC architecture from Section 3.4.1 / Figure
3.3 of the proposal:

    View (HTML/JS)  <-->  Controller (this file)  <-->  Model (Conformer)

It does three things:
    1. Serves the site (index/analysis/results/recommendations/about).
    2. Exposes POST /api/analyze - accepts the audio recorded on the
       Real-Time Analysis page, runs it through the SAME preprocessing +
       feature extraction + trained Conformer model used in train.py /
       evaluate.py, and returns the real quality_label + confidence.
    3. Falls back to a clearly-labeled placeholder response if the trained
       model file isn't found (e.g. you haven't run train.py yet), so the
       front-end still works end-to-end during development.

Run:
    pip install flask pydub imageio-ffmpeg
    python main.py
Then open http://127.0.0.1:5000/

NOTE on pydub/imageio-ffmpeg: browsers record microphone audio as
WebM/Opus (MediaRecorder's default), not WAV. imageio-ffmpeg bundles an
ffmpeg binary directly (no manual install/PATH setup needed on Windows) so
pydub can convert the uploaded WebM clip to WAV before it's fed into
preprocess_audio().
"""

import os
import tempfile

import numpy as np
from flask import Flask, render_template, jsonify, request

from preprocessing import preprocess_audio, TARGET_SR
from features import extract_mfcc, extract_pitch, detect_pauses

# Importing conformer_model triggers its @register_keras_serializable
# decorators - required before tf.keras.models.load_model() can load a
# saved model that uses the custom ConvSubsampling/ConformerBlock/etc.
# layers (same reason evaluate.py imports it).
import conformer_model  # noqa: F401
from train import LABEL_MAP, MAX_FRAMES, pad_or_truncate

import tensorflow as tf

# Your HTML/CSS/JS live flat in ../frontend (not split into static/templates
# subfolders), so both static serving and template rendering point there.
# static_url_path="" means frontend/style.css is served at /style.css and
# frontend/script.js at /script.js, matching the HTML's <link>/<script> tags.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
    template_folder=FRONTEND_DIR,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation_conformer.keras")
LABEL_NAMES = {idx: name for name, idx in LABEL_MAP.items()}  # 0->poor, 1->average, 2->good

_model = None  # loaded lazily on first request, not at import time


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")


@app.route("/analysis.html")
def analysis():
    return render_template("analysis.html")


@app.route("/results.html")
def results():
    return render_template("results.html")


@app.route("/recommendations.html")
def recommendations():
    return render_template("recommendations.html")


@app.route("/about.html")
def about():
    return render_template("about.html")


# ---------------------------------------------------------------------------
# API - integration point for the Conformer model
# ---------------------------------------------------------------------------

def get_model():
    """Loads the trained .keras model once and caches it in _model.
    Returns None if no trained model file exists yet (dev-mode fallback)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        custom_objects = {
            "ConvSubsampling": conformer_model.ConvSubsampling,
            "FeedForwardModule": conformer_model.FeedForwardModule,
            "MHSAModule": conformer_model.MHSAModule,
            "ConvolutionModule": conformer_model.ConvolutionModule,
            "ConformerBlock": conformer_model.ConformerBlock,
        }
        print(f"Loading trained model from {MODEL_PATH} ...")
        _model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    return _model


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Converts whatever the browser recorded (WebM/Opus, or Ogg on
    Firefox) into a mono 16kHz WAV file, using pydub + a bundled ffmpeg
    binary (imageio-ffmpeg) - no manual ffmpeg install/PATH setup needed."""
    from pydub import AudioSegment
    import imageio_ffmpeg

    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    audio.export(output_path, format="wav")


def run_model_prediction(wav_path: str) -> dict:
    """Runs the SAME preprocessing + feature extraction used during
    training, then the trained Conformer, on one recorded clip."""
    model = get_model()
    if model is None:
        return {
            "status": "placeholder",
            "message": f"No trained model found at {MODEL_PATH} - run train.py first.",
            "quality_label": "average",
            "confidence": 0.0,
        }

    audio, sr = preprocess_audio(wav_path, target_sr=TARGET_SR)
    duration_sec = len(audio) / sr

    mfcc = extract_mfcc(audio, sr)
    x = pad_or_truncate(mfcc, MAX_FRAMES)
    x = np.expand_dims(x, axis=0).astype(np.float32)  # add batch dim -> (1, 1200, 40)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    pitch = extract_pitch(audio, sr)
    pauses = detect_pauses(audio, sr)
    pause_per_min = (pauses["pause_count"] / duration_sec) * 60 if duration_sec > 0 else 0.0

    return {
        "status": "ok",
        "quality_label": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "duration_sec": round(duration_sec, 2),
        "pitch_mean_hz": round(pitch["mean_pitch_hz"], 1),
        "pitch_std_hz": round(pitch["pitch_std_hz"], 1),
        "pause_count": pauses["pause_count"],
        "pause_per_min": round(pause_per_min, 1),
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Expects a multipart/form-data POST with an audio file field named
    'audio' (the Real-Time Analysis page uploads the recorded session
    clip here when the user clicks Stop - see script.js). Runs the real
    trained Conformer model and returns its prediction.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided under the 'audio' field."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    raw_fd, raw_path = tempfile.mkstemp(suffix=os.path.splitext(audio_file.filename)[1] or ".webm")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(raw_fd)
    os.close(wav_fd)

    try:
        audio_file.save(raw_path)
        convert_to_wav(raw_path, wav_path)
        result = run_model_prediction(wav_path)
        status_code = 200 if result["status"] == "ok" else 200  # placeholder is still a valid response
        return jsonify(result), status_code
    except Exception as e:
        app.logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    finally:
        for p in (raw_path, wav_path):
            try:
                os.remove(p)
            except OSError:
                pass


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
