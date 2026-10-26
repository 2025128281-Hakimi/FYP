import json
import os
import tempfile

import numpy as np
from flask import Flask, render_template, jsonify, request

from preprocessing import preprocess_audio, load_audio, TARGET_SR
from features import extract_mfcc
from filler_detector import detect_fillers, WINDOW_FRAMES, N_MELS

import conformer_model
from train import LABEL_MAP, MAX_FRAMES, pad_or_truncate

import tensorflow as tf

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
    template_folder=FRONTEND_DIR,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation_conformer.keras")
FILLER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filler_detector.keras")
LABEL_NAMES = {idx: name for name, idx in LABEL_MAP.items()}

_model = None
_filler_model = None


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


def get_model():
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
        try:
            # The first predict() is slow while TensorFlow prepares the model,
            # so do it once here instead of on the user's first analysis.
            warmup = np.zeros((1, MAX_FRAMES, _model.input_shape[-1]), dtype=np.float32)
            _model.predict(warmup, verbose=0)
        except Exception as e:
            print(f"Model warm-up skipped: {e}")
    return _model


def get_filler_model():
    global _filler_model
    if _filler_model is None:
        if not os.path.exists(FILLER_MODEL_PATH):
            return None
        print(f"Loading filler detector from {FILLER_MODEL_PATH} ...")
        _filler_model = tf.keras.models.load_model(FILLER_MODEL_PATH)
        try:
            _filler_model.predict(np.zeros((1, WINDOW_FRAMES, N_MELS, 1), dtype=np.float32), verbose=0)
        except Exception as e:
            print(f"Filler detector warm-up skipped: {e}")
    return _filler_model


def run_filler_detection(wav_path: str) -> dict:
    model = get_filler_model()
    if model is None:
        return {}
    # Raw audio (no noise reduction/trimming), matching the clips the detector was trained on.
    audio, _ = load_audio(wav_path, TARGET_SR)
    return detect_fillers(audio, model, **get_filler_rules())


def get_filler_rules() -> dict:
    rules = {"threshold": 0.5, "min_windows": 2}
    path = os.path.splitext(FILLER_MODEL_PATH)[0] + "_threshold.json"
    try:
        with open(path) as f:
            saved = json.load(f)
        rules["threshold"] = float(saved.get("threshold", rules["threshold"]))
        rules["min_windows"] = int(saved.get("min_windows", rules["min_windows"]))
    except (OSError, ValueError):
        pass
    return rules


def convert_to_wav(input_path: str, output_path: str) -> None:
    # Call ffmpeg directly: pydub also needs ffprobe, which imageio_ffmpeg
    # doesn't ship, so browser webm recordings failed to convert.
    import subprocess
    import imageio_ffmpeg

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-i", input_path, "-ar", str(TARGET_SR), "-ac", "1", output_path],
        check=True,
        capture_output=True,
    )


def run_model_prediction(wav_path: str) -> dict:
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
    x = np.expand_dims(x, axis=0).astype(np.float32)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "status": "ok",
        "quality_label": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "duration_sec": round(duration_sec, 2),
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
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
        try:
            result.update(run_filler_detection(wav_path))
        except Exception:
            app.logger.exception("Filler detection failed - falling back to the browser's count")
        status_code = 200 if result["status"] == "ok" else 200
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
    get_model()
    get_filler_model()
    app.run(debug=True, use_reloader=False)
