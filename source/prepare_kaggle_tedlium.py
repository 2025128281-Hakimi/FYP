import argparse
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf

from preprocessing import load_audio, reduce_noise, trim_silence, normalize, TARGET_SR
from features import extract_pitch, detect_pauses
from rubric import label_from_features
from labeling_io import finalize

FILLER_PATTERN = re.compile(r'\{\s*(uh|um|ah|er|erm|uhh|umm)\s*\}', re.IGNORECASE)
TAG_PATTERN = re.compile(r'\{[^}]*\}')
ANGLE_PATTERN = re.compile(r'<[^>]*>')
PAREN_NUM_PATTERN = re.compile(r'\(\d+\)')

CLIP_ID_PATTERN = re.compile(r'^(.*)_Segment(\d+)$', re.IGNORECASE)


def clean_transcript_and_count(text: str) -> tuple:
    filler_count = len(FILLER_PATTERN.findall(text))
    cleaned = TAG_PATTERN.sub(' ', text)
    cleaned = ANGLE_PATTERN.sub(' ', cleaned)
    cleaned = PAREN_NUM_PATTERN.sub('', cleaned)
    words = [w for w in cleaned.strip().split() if w]
    return len(words), filler_count


def parse_clip_id(clip_id: str):
    m = CLIP_ID_PATTERN.match(clip_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def build_windows(talk_segments: list, window_sec: float, min_window_sec: float) -> list:
    windows = []
    current = []
    current_duration = 0.0

    for seg in talk_segments:
        current.append(seg)
        current_duration += seg["duration_sec"]
        if current_duration >= window_sec:
            windows.append(current)
            current = []
            current_duration = 0.0

    if current:
        leftover_duration = sum(s["duration_sec"] for s in current)
        if leftover_duration >= min_window_sec or not windows:
            windows.append(current)
        else:
            windows[-1].extend(current)

    return windows


def process_window(talk_id: str, window_idx: int, segments: list, clips_dir: str) -> dict:
    audio = np.concatenate([s["audio"] for s in segments])
    sr = segments[0]["sr"]
    transcript = " ".join(s["transcript"] for s in segments)

    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio)
    audio = normalize(audio)

    duration_sec = len(audio) / sr
    duration_min = max(duration_sec / 60.0, 1e-6)

    pitch = extract_pitch(audio, sr)
    pauses = detect_pauses(audio, sr)
    word_count, filler_count = clean_transcript_and_count(transcript)

    feats = {
        "duration_sec": round(duration_sec, 1),
        "pitch_std_hz": round(pitch["pitch_std_hz"], 1),
        "pause_per_min": round(pauses["pause_count"] / duration_min, 2),
        "speech_rate_wpm": round(word_count / duration_min, 1),
        "filler_per_min": round(filler_count / duration_min, 2),
    }

    label, scores = label_from_features(feats)

    clip_id = f"{talk_id}_win{window_idx:02d}"
    clip_path = os.path.join(clips_dir, clip_id + ".wav")
    sf.write(clip_path, audio, sr)

    return {"clip_id": clip_id, "path": clip_path, "label": label,
            "source": "tedlium1_kaggle", "n_segments": len(segments), **feats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                         help='Path to the "archive (10)" folder (contains train/ and test/)')
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output_dir", required=True,
                         help="Where to write good/average/poor subfolders (matches train.py's --data_dir)")
    parser.add_argument("--report", default="kaggle_tedlium_label_report.csv")
    parser.add_argument("--sample_for_review", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap the number of RAW SEGMENTS read from the parquet before windowing - "
                              "useful for a quick first test. The resulting number of windows will be "
                              "smaller than this, since several segments merge into each window.")
    parser.add_argument("--window_sec", type=float, default=20.0,
                         help="Target minimum length of each reassembled window, in seconds")
    parser.add_argument("--min_window_sec", type=float, default=10.0,
                         help="A trailing leftover shorter than this gets merged into the previous "
                              "window instead of standing alone")
    args = parser.parse_args()

    parquet_path = os.path.join(args.dataset_root, args.split, f"{args.split}_transcript.parquet")
    wav_dir = os.path.join(args.dataset_root, args.split, "wav")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} not found - check --dataset_root")
    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"{wav_dir} not found - check --dataset_root")

    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} rows found (columns: {df.columns.tolist()})")

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"  Capped to first {len(df)} rows via --max_samples")

    talks = {}
    skipped_missing = 0
    skipped_unparsed = 0
    skipped_error = 0

    for i, row in df.iterrows():
        clip_id = row["File"]
        talk_id, segment_num = parse_clip_id(clip_id)
        if talk_id is None:
            skipped_unparsed += 1
            continue

        wav_path = os.path.join(wav_dir, clip_id + ".wav")
        if not os.path.exists(wav_path):
            skipped_missing += 1
            continue

        try:
            audio, sr = load_audio(wav_path, TARGET_SR)
            talks.setdefault(talk_id, []).append({
                "segment_num": segment_num,
                "clip_id": clip_id,
                "audio": audio,
                "sr": sr,
                "duration_sec": len(audio) / sr,
                "transcript": str(row["Transcript"]),
            })
        except Exception as e:
            skipped_error += 1
            print(f"  {clip_id}: FAILED to load ({e}) - skipping")
            continue

    print(f"\nLoaded {sum(len(v) for v in talks.values())} segments across {len(talks)} talks.")
    print(f"Skipped: {skipped_missing} missing wav, {skipped_unparsed} unparsed clip_id, "
          f"{skipped_error} load errors")

    rows = []
    for talk_id, segments in talks.items():
        segments.sort(key=lambda s: s["segment_num"])
        windows = build_windows(segments, args.window_sec, args.min_window_sec)

        for window_idx, window_segments in enumerate(windows):
            try:
                clips_dir = os.path.join(args.output_dir, "_raw_kaggle_tedlium_windows")
                os.makedirs(clips_dir, exist_ok=True)
                row = process_window(talk_id, window_idx, window_segments, clips_dir)
                rows.append(row)
            except Exception as e:
                print(f"  {talk_id} window {window_idx}: FAILED ({e}) - skipping")
                continue

        if len(rows) % 20 < len(windows):
            print(f"  ...processed through talk '{talk_id}' ({len(rows)} windows so far)")

    print(f"\nBuilt {len(rows)} windows from {len(talks)} talks "
          f"(avg {sum(len(v) for v in talks.values()) / max(len(talks), 1):.1f} raw segments/talk).")

    finalize(rows, args.output_dir, args.report, args.dry_run, args.sample_for_review)


if __name__ == "__main__":
    main()
