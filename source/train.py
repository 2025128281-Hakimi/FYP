import argparse
import csv
import glob
import os
import re

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import TARGET_SR
from features import extract_mfcc
from conformer_model import build_conformer_model

LABEL_MAP = {"poor": 0, "average": 1, "good": 2}
LABEL_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]


MAX_FRAMES = 1200

WINDOW_SUFFIX = re.compile(r"_win\d+$")


def pad_or_truncate(mfcc: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    t, f = mfcc.shape
    if t >= max_frames:
        return mfcc[:max_frames]
    pad = np.zeros((max_frames - t, f), dtype=mfcc.dtype)
    return np.vstack([mfcc, pad])


def load_labeled_dataset(data_dir: str, cache_path: str = None, force_refresh: bool = False,
                         return_files: bool = False):
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached features from {cache_path} ...")
        cached = np.load(cache_path)
        if not return_files:
            return cached["X"], cached["y"]
        if "files" in cached:
            return cached["X"], cached["y"], cached["files"]
        print("  (old cache without file names - re-extracting)")

    X, y, files = [], [], []
    for label_name, label_idx in LABEL_MAP.items():
        pattern = os.path.join(data_dir, label_name, "*.wav")
        paths = sorted(glob.glob(pattern))
        print(f"  {label_name}: {len(paths)} files")
        for n, path in enumerate(paths, 1):
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:

                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            mfcc = extract_mfcc(audio.astype(np.float32), sr)
            X.append(pad_or_truncate(mfcc).astype(np.float32))
            y.append(label_idx)
            files.append(os.path.splitext(os.path.basename(path))[0])
            if n % 1000 == 0:
                print(f"    {n}/{len(paths)}")

    if not X:
        raise ValueError(
            f"No .wav files found under {data_dir}/<good|average|poor>/. "
            "Populate this folder before training - see load_labeled_dataset() docstring."
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    files = np.array(files)

    if cache_path:
        np.savez(cache_path, X=X, y=y, files=files)
        print(f"Cached extracted features to {cache_path}")

    if return_files:
        return X, y, files
    return X, y


def split_by_talk(files: np.ndarray, val_split: float, test_split: float, seed: int):
    """Split so every window of a talk lands in the same set. A random per-clip split
    would put windows of the same speaker in both train and test, which inflates the scores."""
    talks = np.array([WINDOW_SUFFIX.sub("", f) for f in files])
    unique = np.unique(talks)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_test = int(len(unique) * test_split)
    n_val = int(len(unique) * val_split)
    test_talks = set(unique[:n_test])
    val_talks = set(unique[n_test:n_test + n_val])
    test_idx = np.where([t in test_talks for t in talks])[0]
    val_idx = np.where([t in val_talks for t in talks])[0]
    train_idx = np.where([t not in test_talks and t not in val_talks for t in talks])[0]
    return train_idx, val_idx, test_idx, len(unique)


class EpochMetrics(tf.keras.callbacks.Callback):
    """Scores the validation set after every epoch (loss, accuracy, per-class
    precision/recall/F1, macro F1) and appends a row to a CSV."""

    def __init__(self, X_val, y_val, csv_path, batch_size):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probs = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        pred = np.argmax(probs, axis=1)
        p_true = np.clip(probs[np.arange(len(self.y_val)), self.y_val], 1e-7, 1.0)
        p, r, f1, _ = precision_recall_fscore_support(
            self.y_val, pred, labels=list(range(len(LABEL_NAMES))), zero_division=0)

        row = {
            "epoch": epoch + 1,
            "train_loss": round(float(logs.get("loss", np.nan)), 4),
            "train_accuracy": round(float(logs.get("accuracy", np.nan)), 4),
            "val_loss": round(float(-np.mean(np.log(p_true))), 4),
            "val_accuracy": round(float(np.mean(pred == self.y_val)), 4),
            "val_macro_f1": round(float(np.mean(f1)), 4),
        }
        for i, name in enumerate(LABEL_NAMES):
            row[f"{name}_precision"] = round(float(p[i]), 4)
            row[f"{name}_recall"] = round(float(r[i]), 4)
            row[f"{name}_f1"] = round(float(f1[i]), 4)
        self.rows.append(row)

        logs["val_loss"] = row["val_loss"]
        logs["val_accuracy"] = row["val_accuracy"]
        logs["val_macro_f1"] = row["val_macro_f1"]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"  val: loss {row['val_loss']:.4f} | acc {row['val_accuracy']:.4f} | "
              f"macro F1 {row['val_macro_f1']:.4f} | "
              + " | ".join(f"{n} F1 {row[f'{n}_f1']:.2f}" for n in LABEL_NAMES))


def plot_curves(rows: list, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping the training curves image)")
        return

    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [r["train_accuracy"] for r in rows], label="train")
    axes[1].plot(epochs, [r["val_accuracy"] for r in rows], label="validation")
    axes[1].set_title("Accuracy")
    axes[2].plot(epochs, [r["val_macro_f1"] for r in rows], label="macro F1", linewidth=2)
    for name in LABEL_NAMES:
        axes[2].plot(epochs, [r[f"{name}_f1"] for r in rows], label=f"{name} F1", alpha=0.7)
    axes[2].set_title("Validation F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder with good/average/poor subfolders of .wav clips")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--out", type=str, default="presentation_conformer.keras")
    parser.add_argument("--force_refresh", action="store_true",
                         help="Ignore any cached features and re-extract from the .wav files")
    parser.add_argument("--no_early_stop", action="store_true",
                         help="Epoch experiment: run every epoch instead of stopping when the "
                              "validation macro F1 stops improving")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None,
                         help="Train on a random subset of this many clips (quick test)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for the train/val/test split, so it's reproducible - "
                              "evaluate.py needs the EXACT same split to report honest metrics "
                              "on truly held-out data, not clips the model already trained on.")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    cache_path = os.path.join(args.data_dir, "_features_cache.npz")
    print("Loading and featurizing dataset...")
    X, y, files = load_labeled_dataset(args.data_dir, cache_path=cache_path,
                                       force_refresh=args.force_refresh, return_files=True)
    print(f"Total samples: {len(X)}  |  feature shape: {X.shape[1:]}")

    train_idx, val_idx, test_idx, n_talks = split_by_talk(files, args.val_split, args.test_split, args.seed)
    if args.limit:
        rng = np.random.RandomState(args.seed)
        scale = args.limit / len(X)
        train_idx, val_idx, test_idx = (
            np.sort(rng.choice(i, max(1, int(len(i) * scale)), replace=False))
            for i in (train_idx, val_idx, test_idx))
        print(f"  --limit: using {len(train_idx) + len(val_idx) + len(test_idx)} clips")

    split_path = args.out + ".split.npz"
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=args.seed)
    print(f"Split by talk ({n_talks} talks): train {len(train_idx)} | val {len(val_idx)} | "
          f"test {len(test_idx)} clips. Saved to {split_path} (for evaluate.py)")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    weights = compute_class_weight("balanced", classes=np.arange(len(LABEL_NAMES)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Class weights: " + ", ".join(f"{LABEL_NAMES[i]} {w:.2f}" for i, w in class_weight.items()))

    # MFCC values range from about -600 to +200; the model learns far better on
    # standardised input. The layer is saved inside the model, so main.py needs no change.
    normalizer = tf.keras.layers.Normalization(axis=-1, name="mfcc_norm")
    normalizer.adapt(X_train.reshape(-1, X.shape[-1])[::50])

    model = build_conformer_model(
        input_shape=X.shape[1:], num_classes=len(LABEL_MAP), normalizer=normalizer
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    base = os.path.splitext(args.out)[0]
    metrics_cb = EpochMetrics(X_val, y_val, base + "_epochs.csv", args.batch_size)
    callbacks = [
        metrics_cb,
        tf.keras.callbacks.ModelCheckpoint(args.out, monitor="val_macro_f1", mode="max",
                                           save_best_only=True),
    ]
    if not args.no_early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=args.patience))

    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    rows = metrics_cb.rows
    best = max(rows, key=lambda r: r["val_macro_f1"])
    print(f"\nPer-epoch results saved to {base}_epochs.csv")
    print("\nEpoch  val_acc  macro_F1  " + "  ".join(f"{n}_F1" for n in LABEL_NAMES))
    for r in rows:
        if r["epoch"] % 10 == 0 or r is best:
            print(f"{r['epoch']:>5}  {r['val_accuracy']:.4f}   {r['val_macro_f1']:.4f}  "
                  + "  ".join(f"{r[f'{n}_f1']:>{len(n) + 3}.4f}" for n in LABEL_NAMES)
                  + ("   <- best" if r is best else ""))
    plot_curves(rows, base + "_curves.png")
    print(f"Best epoch: {best['epoch']} (val macro F1 {best['val_macro_f1']:.4f}, "
          f"val accuracy {best['val_accuracy']:.4f}). That model is saved to {args.out}")
    print("Next: run evaluate.py to score it on the unseen test talks.")


if __name__ == "__main__":
    main()
