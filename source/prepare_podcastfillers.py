import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from features import extract_pitch, detect_pauses
from rubric import label_from_features
from labeling_io import finalize

FILLER_LABELS = {"Uh", "Um", "You know", "Like", "Other"}


def load_filler_events(csv_path: str) -> dict:
    events_by_episode = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label_full_vocab"] in FILLER_LABELS:
                episode = row["podcast_filename"]
                start = float(row["event_start_inepisode"])
                events_by_episode[episode].append(start)
    return events_by_episode


def load_transcript_word_times(transcript_path: str) -> list:
    if not os.path.exists(transcript_path):
        return []
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_times = []
    for seg in data.get("segments", []):
        nbest = seg.get("nbest", [])
        if not nbest:
            continue
        for w in nbest[0].get("words", []):
            if "offset" in w:
                word_times.append(w["offset"] / 1e7)
    return word_times


def find_episode_wav(dataset_root: str, episode: str) -> str | None:
    for split in ("train", "validation", "test"):
        candidate = os.path.join(dataset_root, "audio", "episode_wav", split, episode + ".wav")
        if os.path.exists(candidate):
            return candidate
    return None


def find_transcript_json(dataset_root: str, episode: str) -> str | None:
    for split in ("train", "validation", "test"):
        candidate = os.path.join(dataset_root, "metadata", "episode_transcripts", split, episode + ".json")
        if os.path.exists(candidate):
            return candidate
    return None


def chunk_episode(dataset_root: str, episode: str, window_sec: float,
                   filler_times: list, word_times: list, clips_dir: str,
                   clip_prefix: str) -> list:
    wav_path = find_episode_wav(dataset_root, episode)
    if wav_path is None:
        print(f"  Skipping {episode}: episode wav not found")
        return []

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_duration = len(audio) / sr

    rows = []
    n_windows = int(total_duration // window_sec)
    window_min = window_sec / 60.0

    for w in range(n_windows):
        w_start = w * window_sec
        w_end = w_start + window_sec

        start_sample = int(w_start * sr)
        end_sample = int(w_end * sr)
        window_audio = audio[start_sample:end_sample]

        peak = np.max(np.abs(window_audio)) or 1.0
        window_audio = (window_audio / peak).astype(np.float32)

        pitch = extract_pitch(window_audio, sr)
        pauses = detect_pauses(window_audio, sr)

        filler_count = sum(1 for t in filler_times if w_start <= t < w_end)
        word_count = sum(1 for t in word_times if w_start <= t < w_end)

        feats = {
            "duration_sec": round(window_sec, 1),
            "pitch_std_hz": round(pitch["pitch_std_hz"], 1),
            "pause_per_min": round(pauses["pause_count"] / window_min, 2),
            "speech_rate_wpm": round(word_count / window_min, 1),
            "filler_per_min": round(filler_count / window_min, 2),
        }

        label, scores = label_from_features(feats)

        clip_id = f"{clip_prefix}_{w:04d}"
        clip_path = os.path.join(clips_dir, clip_id + ".wav")
        sf.write(clip_path, window_audio, sr)

        rows.append({"clip_id": clip_id, "path": clip_path, "label": label,
                     "source": "podcastfillers", **feats})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                         help="Path to the extracted PodcastFillers folder (contains metadata/ and audio/)")
    parser.add_argument("--output_dir", required=True,
                         help="Where to write good/average/poor subfolders (matches train.py's --data_dir)")
    parser.add_argument("--window_sec", type=float, default=20.0,
                         help="Length of each chunked window, in seconds")
    parser.add_argument("--max_episodes", type=int, default=15,
                         help="How many episodes to process (laptop-scale default - each episode yields dozens of windows)")
    parser.add_argument("--report", default="podcastfillers_label_report.csv")
    parser.add_argument("--sample_for_review", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    csv_path = os.path.join(args.dataset_root, "metadata", "PodcastFillers.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found - check --dataset_root points at the extracted PodcastFillers folder."
        )

    print("Loading filler event annotations...")
    filler_events = load_filler_events(csv_path)
    episodes = list(filler_events.keys())[: args.max_episodes]
    print(f"Processing {len(episodes)} of {len(filler_events)} total episodes...")

    clips_dir = os.path.join(args.output_dir, "_raw_podcastfillers_clips")
    os.makedirs(clips_dir, exist_ok=True)

    all_rows = []
    for i, episode in enumerate(episodes, 1):
        print(f"  [{i}/{len(episodes)}] {episode}")
        transcript_path = find_transcript_json(args.dataset_root, episode)
        word_times = load_transcript_word_times(transcript_path) if transcript_path else []
        if transcript_path is None:
            print(f"      (no transcript json found for this episode - speech_rate_wpm will be 0)")

        rows = chunk_episode(
            args.dataset_root, episode, args.window_sec,
            filler_events[episode], word_times, clips_dir,
            clip_prefix=f"pf_{i:03d}",
        )
        print(f"      -> {len(rows)} windows")
        all_rows.extend(rows)

    finalize(all_rows, args.output_dir, args.report, args.dry_run, args.sample_for_review)


if __name__ == "__main__":
    main()
