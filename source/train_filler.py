import argparse
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf

from filler_detector import SR, clip_features, build_filler_model

FILLER_LABELS = {"Uh", "Um"}
SPLITS = ("train", "validation", "test")


def load_split(dataset_root: str, df: pd.DataFrame, split: str, limit: int = None):
    rows = df[df["clip_split_subset"] == split]
    if limit:
        rows = rows.sample(n=min(limit, len(rows)), random_state=42)

    X, y = [], []
    clip_dir = os.path.join(dataset_root, "audio", "clip_wav", split)
    for i, r in enumerate(rows.itertuples(), 1):
        path = os.path.join(clip_dir, r.clip_name)
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception as e:
            print(f"  {path}: FAILED to read ({e}) - skipping")
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        X.append(clip_features(audio).astype(np.float16))
        y.append(1 if r.label_consolidated_vocab in FILLER_LABELS else 0)
        if i % 5000 == 0:
            print(f"  {split}: {i}/{len(rows)} clips")

    return np.stack(X)[..., np.newaxis], np.array(y, dtype=np.float32)


def load_dataset(dataset_root: str, cache_path: str, force_refresh: bool, limit: int = None):
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached features from {cache_path} ...")
        data = np.load(cache_path)
        return {s: (data[f"X_{s}"], data[f"y_{s}"]) for s in SPLITS}

    csv_path = os.path.join(dataset_root, "metadata", "PodcastFillers.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found - check --dataset_root")
    df = pd.read_csv(csv_path)
    df = df[df["label_consolidated_vocab"].notna()]

    out = {}
    for split in SPLITS:
        print(f"Featurizing {split} clips...")
        out[split] = load_split(dataset_root, df, split, limit)

    if cache_path:
        np.savez(cache_path, **{f"X_{s}": out[s][0] for s in SPLITS},
                 **{f"y_{s}": out[s][1] for s in SPLITS})
        print(f"Cached features to {cache_path}")
    return out


def scores(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-10),
    }


def best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    candidates = np.arange(0.10, 0.91, 0.05)
    return float(max(candidates, key=lambda t: scores(y_true, probs, t)["f1"]))


def report(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> None:
    s = scores(y_true, probs, threshold)
    tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
    accuracy, precision, recall, f1 = s["accuracy"], s["precision"], s["recall"], s["f1"]
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}  (of clips flagged as filler, how many really were)")
    print(f"  Recall:    {recall:.3f}  (of real fillers, how many were caught)")
    print(f"  F1-score:  {f1:.3f}")
    print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                         help="Extracted PodcastFillers folder (contains audio/ and metadata/)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="filler_detector.keras")
    parser.add_argument("--cache", default="filler_features_cache.npz",
                         help="Where to cache extracted features (re-used on the next run)")
    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="Use at most N clips per split - for a quick test run")
    args = parser.parse_args()

    data = load_dataset(args.dataset_root, args.cache, args.force_refresh, args.limit)
    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]
    X_test, y_test = data["test"]
    for name, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
        print(f"{name}: {len(y)} clips ({int(y.sum())} fillers, {int(len(y) - y.sum())} other)")

    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weight = {0: len(y_train) / (2 * neg), 1: len(y_train) / (2 * pos)}

    model = build_filler_model(X_train.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Pick the decision threshold on the validation set (not the test set, so the
    # test results below stay an honest measure on unseen clips).
    threshold = best_threshold(y_val, model.predict(X_val, batch_size=256, verbose=0)[:, 0])
    threshold_path = os.path.splitext(args.out)[0] + "_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": round(threshold, 2)}, f)
    print(f"\nBest threshold on the validation set: {threshold:.2f} (saved to {threshold_path})")

    print("\nTest set results (clips the model never saw during training):")
    report(y_test, model.predict(X_test, batch_size=256, verbose=0)[:, 0], threshold)
    print(f"\nBest model saved to {args.out} - restart main.py to use it.")


if __name__ == "__main__":
    main()
