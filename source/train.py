import argparse
import glob
import os

import numpy as np
import soundfile as sf
import tensorflow as tf

from preprocessing import TARGET_SR
from features import extract_mfcc
from conformer_model import build_conformer_model

LABEL_MAP = {"poor": 0, "average": 1, "good": 2}


MAX_FRAMES = 1200


def pad_or_truncate(mfcc: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    t, f = mfcc.shape
    if t >= max_frames:
        return mfcc[:max_frames]
    pad = np.zeros((max_frames - t, f), dtype=mfcc.dtype)
    return np.vstack([mfcc, pad])


def load_labeled_dataset(data_dir: str, cache_path: str = None, force_refresh: bool = False):
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached features from {cache_path} ...")
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    X, y = [], []
    for label_name, label_idx in LABEL_MAP.items():
        pattern = os.path.join(data_dir, label_name, "*.wav")
        files = glob.glob(pattern)
        print(f"  {label_name}: {len(files)} files")
        for path in files:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:

                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            mfcc = extract_mfcc(audio.astype(np.float32), sr)
            X.append(pad_or_truncate(mfcc))
            y.append(label_idx)

    if not X:
        raise ValueError(
            f"No .wav files found under {data_dir}/<good|average|poor>/. "
            "Populate this folder before training - see load_labeled_dataset() docstring."
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if cache_path:
        np.savez(cache_path, X=X, y=y)
        print(f"Cached extracted features to {cache_path}")

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder with good/average/poor subfolders of .wav clips")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="presentation_conformer.keras")
    parser.add_argument("--force_refresh", action="store_true",
                         help="Ignore any cached features and re-extract from the .wav files")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for the train/val split, so it's reproducible - "
                              "evaluate.py needs the EXACT same split to report honest metrics "
                              "on truly held-out data, not clips the model already trained on.")
    args = parser.parse_args()

    cache_path = os.path.join(args.data_dir, "_features_cache.npz")
    print("Loading and featurizing dataset...")
    X, y = load_labeled_dataset(args.data_dir, cache_path=cache_path, force_refresh=args.force_refresh)
    print(f"Total samples: {len(X)}  |  feature shape: {X.shape[1:]}")


    rng = np.random.RandomState(args.seed)
    n_val = int(len(X) * args.val_split)
    idx = rng.permutation(len(X))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    split_path = args.out + ".split.npz"
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, seed=args.seed)
    print(f"Saved train/val split indices to {split_path} (for evaluate.py)")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = build_conformer_model(
        input_shape=X.shape[1:], num_classes=len(LABEL_MAP)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    print(f"Training complete. Best model saved to {args.out}")
    print("Next: run evaluate.py (accuracy/precision/recall/F1/confusion matrix - Table 3.11)")


if __name__ == "__main__":
    main()
