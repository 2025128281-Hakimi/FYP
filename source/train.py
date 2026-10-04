"""
train.py
--------
Training skeleton that connects: datasets -> preprocessing -> features
-> Conformer model -> training loop.

This is deliberately a SKELETON, not a finished pipeline, because the
labeling step (presentation quality: good/average/poor) needs a decision
from you first - see the "LABELING" section below and our earlier dataset
discussion.

Two-stage plan:
  Stage A - Pretrain/adapt the Conformer as a speech encoder on TED-LIUM
            (unsupervised or ASR-style objective), OR skip this and use
            it directly as a supervised classifier if your timeline is tight.
  Stage B - Fine-tune / train the classification head on a smaller,
            labeled set (MIT Interview Dataset, POM, or your own
            self-labeled TED clips) to predict quality: good/average/poor.

Run:
    python train.py --data_dir ./data/labeled_clips --epochs 20
"""

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

# MAX_FRAMES was originally 400, which at the default MFCC hop length
# (512 samples @ 16kHz -> ~31.25 frames/sec) covers only ~12.8 SECONDS of
# audio - but our labeled windows run ~20s on average and up to 37.1s
# (confirmed via combined_label_report.csv's duration_sec column: mean
# 20.6s, max 37.1s). At 400 frames, every clip longer than ~13s was
# silently truncated, throwing away the back half or more of most clips
# before the model ever saw it. 1200 frames (~38.4s) covers the real max
# with a small safety margin.
MAX_FRAMES = 1200


def pad_or_truncate(mfcc: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """MFCC sequences vary in length; pad/truncate so they can be batched."""
    t, f = mfcc.shape
    if t >= max_frames:
        return mfcc[:max_frames]
    pad = np.zeros((max_frames - t, f), dtype=mfcc.dtype)
    return np.vstack([mfcc, pad])


def load_labeled_dataset(data_dir: str, cache_path: str = None, force_refresh: bool = False):
    """
    Expects a folder structure like:
        data_dir/
            good/*.wav
            average/*.wav
            poor/*.wav

    This matches the combined output of merge_datasets.py (TED-LIUM +
    PodcastFillers clips, already labeled by the rubric).

    NOTE: unlike the original version, this does NOT re-run the full
    preprocess_audio() pipeline (noise reduction + silence trimming +
    normalization) on each clip. Every clip here already went through
    that exact pipeline once during labeling (prepare_kaggle_tedlium.py /
    prepare_podcastfillers.py both call reduce_noise/trim_silence/
    normalize before writing the .wav). Re-running noise reduction a
    second time on already-clean audio does nothing useful and is one of
    the more expensive steps (spectral gating over every clip) - across
    ~4000 clips that's a lot of wasted laptop time for zero benefit. This
    just loads the already-processed audio directly.

    Also caches the extracted MFCC features to disk (as a .npz) so that
    re-running train.py later (different --epochs, --batch_size, etc.)
    doesn't repeat the full feature-extraction pass over every clip. Pass
    force_refresh=True (--force_refresh on the CLI) if you've relabeled
    or changed the data_dir contents and need the cache rebuilt.
    """
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
                # Shouldn't normally happen (all clips were written at
                # TARGET_SR), but guard against it rather than silently
                # computing MFCCs at the wrong rate.
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

    # Fixed seed (instead of the previous unseeded np.random.permutation) so
    # this split is reproducible - and we ALSO save the exact indices used,
    # so evaluate.py can load this file and evaluate on precisely the same
    # held-out clips this run trained on, rather than guessing a fresh split
    # that might overlap with training data and inflate reported accuracy.
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
