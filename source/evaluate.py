"""
evaluate.py
------------
Evaluates a trained presentation-quality Conformer model and produces the
accuracy / precision / recall / F1 / confusion matrix report your proposal
calls for (Table 3.11).

IMPORTANT: this evaluates on the EXACT same held-out validation split that
train.py used - loaded from the "<model>.split.npz" file train.py saves
alongside the model checkpoint. This is deliberate: without that saved
split, there'd be no way to know which clips the model was actually
trained on vs. held out, and a freshly-reshuffled "validation set" could
silently include clips the model already saw during training - inflating
every metric here without you knowing it.

If you don't have a "<model>.split.npz" file (e.g. it's from an older
run of train.py before this was added), you'll need to re-run train.py
once so it gets created - see train.py's --seed argument.

Usage:
    python evaluate.py --data_dir ../data/labeled_clips_combined \\
        --model presentation_conformer.keras --out_report evaluation_report.csv
"""

import argparse
import os

import numpy as np
import tensorflow as tf

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    raise SystemExit(
        "scikit-learn isn't installed in this environment. Install it first:\n"
        "    pip install scikit-learn\n"
        "(same venv you've been using for everything else)."
    )

from train import LABEL_MAP, load_labeled_dataset

# Loading the saved .keras model requires the custom layer classes
# (ConvSubsampling, ConformerBlock, etc.) to be registered/importable -
# this import alone triggers their @register_keras_serializable
# decorators, without which tf.keras.models.load_model() fails with
# "Could not locate class 'ConvSubsampling'".
import conformer_model  # noqa: F401

# Ordered so index 0/1/2 match LABEL_MAP's poor/average/good -> 0/1/2
LABEL_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]


def load_split(split_path: str):
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. This file is created by train.py alongside the "
            "model checkpoint (train.py --out <model> also writes <model>.split.npz). "
            "If you trained before this was added, re-run train.py once (it'll reuse "
            "the cached features, so it won't be as slow as the first run) to generate it."
        )
    data = np.load(split_path)
    return data["train_idx"], data["val_idx"]


def plot_confusion_matrix(cm: np.ndarray, labels: list, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping the confusion matrix image; "
              "the numbers are still in the CSV/console output. "
              "`pip install matplotlib` if you want the picture too.)")
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix image saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Same --data_dir used for train.py (needs the same "
                              "_features_cache.npz to exist there)")
    parser.add_argument("--model", type=str, default="presentation_conformer.keras")
    parser.add_argument("--out_report", type=str, default="evaluation_report.csv")
    parser.add_argument("--out_confusion_matrix", type=str, default="confusion_matrix.png")
    args = parser.parse_args()

    cache_path = os.path.join(args.data_dir, "_features_cache.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found - run train.py first (even just to build the "
            "feature cache) before evaluating."
        )

    print("Loading cached features...")
    X, y = load_labeled_dataset(args.data_dir, cache_path=cache_path, force_refresh=False)

    split_path = args.model + ".split.npz"
    print(f"Loading train/val split from {split_path}...")
    _, val_idx = load_split(split_path)
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Evaluating on {len(X_val)} held-out validation clips "
          f"(never seen during training).")

    print(f"Loading model from {args.model}...")
    # Explicit custom_objects (on top of the @register_keras_serializable
    # decorators now added to conformer_model.py) - belt-and-suspenders,
    # since this specific .keras file was saved BEFORE those decorators
    # existed, so its embedded config has bare class names ("ConvSubsampling",
    # not a package-qualified registered name). Passing custom_objects
    # keyed by those exact bare names matches regardless.
    custom_objects = {
        "ConvSubsampling": conformer_model.ConvSubsampling,
        "FeedForwardModule": conformer_model.FeedForwardModule,
        "MHSAModule": conformer_model.MHSAModule,
        "ConvolutionModule": conformer_model.ConvolutionModule,
        "ConformerBlock": conformer_model.ConformerBlock,
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print("\n" + classification_report(y_val, y_pred, target_names=LABEL_NAMES, digits=3))

    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(LABEL_NAMES))))
    print("Confusion matrix (rows = true label, columns = predicted label):")
    header = "        " + "  ".join(f"{n:>8}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:>8}" + "  ".join(f"{v:>8}" for v in row))

    # Write the full classification report (per-class + macro/weighted avg) to CSV
    report_dict = classification_report(y_val, y_pred, target_names=LABEL_NAMES,
                                          digits=3, output_dict=True)
    import csv
    with open(args.out_report, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for key in LABEL_NAMES + ["accuracy", "macro avg", "weighted avg"]:
            row = report_dict[key]
            if key == "accuracy":
                writer.writerow(["accuracy", "", "", round(row, 3), ""])
            else:
                writer.writerow([key, round(row["precision"], 3), round(row["recall"], 3),
                                  round(row["f1-score"], 3), int(row["support"])])
    print(f"\nEvaluation report written to {args.out_report}")

    plot_confusion_matrix(cm, LABEL_NAMES, args.out_confusion_matrix)


if __name__ == "__main__":
    main()
