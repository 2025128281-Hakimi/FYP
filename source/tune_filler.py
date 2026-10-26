import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from filler_detector import SR, window_probabilities, find_events

FILLER_LABELS = {"Uh", "Um"}
MATCH_TOLERANCE_SEC = 0.5


def load_segments(dataset_root: str, df: pd.DataFrame, split: str, minutes: float,
                  max_episodes: int) -> list:
    """A `minutes`-long stretch from the middle of each episode in the split,
    with the annotated filler times inside it."""
    segments = []
    episodes = df[df["episode_split_subset"] == split]["podcast_filename"].unique()[:max_episodes]
    for ep in episodes:
        wavs = glob.glob(os.path.join(dataset_root, "audio", "episode_wav", "*", ep + ".wav"))
        if not wavs:
            continue
        events = df[(df["podcast_filename"] == ep) &
                    df["label_consolidated_vocab"].isin(FILLER_LABELS)]["event_start_inepisode"].values
        info = sf.info(wavs[0])
        length = int(minutes * 60 * SR)
        start = max(0, (int(info.duration * SR) - length) // 2)
        audio, sr = sf.read(wavs[0], start=start, frames=length, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            continue
        offset = start / SR
        truth = [t - offset for t in events if offset <= t < offset + len(audio) / SR]
        segments.append({"episode": ep, "audio": audio, "truth": truth})
    return segments


def match(truth: list, detected: list) -> int:
    """One-to-one matches between annotated and detected fillers within the tolerance."""
    used = set()
    hits = 0
    for t in truth:
        best = None
        for j, d in enumerate(detected):
            if j not in used and abs(d - t) <= MATCH_TOLERANCE_SEC:
                if best is None or abs(d - t) < abs(detected[best] - t):
                    best = j
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def evaluate(segments: list, threshold: float, min_windows: int) -> dict:
    n_truth = n_detected = n_hit = 0
    for seg in segments:
        detected = find_events(*seg["probs"], threshold=threshold, min_windows=min_windows)
        n_truth += len(seg["truth"])
        n_detected += len(detected)
        n_hit += match(seg["truth"], detected)
    precision = n_hit / max(n_detected, 1)
    recall = n_hit / max(n_truth, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {"truth": n_truth, "detected": n_detected, "matched": n_hit,
            "precision": precision, "recall": recall, "f1": f1}


def print_result(name: str, r: dict) -> None:
    print(f"  {name}: real fillers {r['truth']}, detected {r['detected']}, matched {r['matched']}")
    print(f"    Precision {r['precision']:.3f} | Recall {r['recall']:.3f} | F1 {r['f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune the filler counting rules on full recordings (validation episodes) "
                    "and report recording-level results on unseen test episodes.")
    parser.add_argument("--dataset_root", required=True,
                         help="Extracted PodcastFillers folder (contains audio/ and metadata/)")
    parser.add_argument("--model", default="filler_detector.keras")
    parser.add_argument("--minutes", type=float, default=5.0,
                         help="Minutes of audio taken from each episode")
    parser.add_argument("--max_episodes", type=int, default=12,
                         help="Episodes used per split")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dataset_root, "metadata", "PodcastFillers.csv"))
    model = tf.keras.models.load_model(args.model)

    data = {}
    for split in ("validation", "test"):
        print(f"Scoring {split} episodes...")
        data[split] = load_segments(args.dataset_root, df, split, args.minutes, args.max_episodes)
        for seg in data[split]:
            seg["probs"] = window_probabilities(seg["audio"], model)
        print(f"  {len(data[split])} episodes, {sum(len(s['truth']) for s in data[split])} annotated fillers")

    print("\nBefore tuning (threshold 0.50, min_windows 2) on test episodes:")
    print_result("test", evaluate(data["test"], 0.5, 2))

    grid = [(t, w) for t in np.arange(0.50, 0.99, 0.05) for w in range(1, 8)]
    threshold, min_windows = max(grid, key=lambda g: evaluate(data["validation"], *g)["f1"])
    print(f"\nBest rules on validation episodes: threshold {threshold:.2f}, min_windows {min_windows}")
    print_result("validation", evaluate(data["validation"], threshold, min_windows))

    print("\nAfter tuning, on test episodes (never used for training or tuning):")
    print_result("test", evaluate(data["test"], threshold, min_windows))

    out = os.path.splitext(args.model)[0] + "_threshold.json"
    with open(out, "w") as f:
        json.dump({"threshold": round(float(threshold), 2), "min_windows": int(min_windows)}, f)
    print(f"\nSaved counting rules to {out} - restart main.py to use them.")


if __name__ == "__main__":
    main()
