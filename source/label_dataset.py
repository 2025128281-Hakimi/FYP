"""
label_dataset.py
-----------------
Runs the rubric (rubric.py) over a folder of audio clips to auto-generate
Good / Average / Poor labels, then sorts the clips into the folder
structure train.py already expects:

    output_dir/
        good/*.wav
        average/*.wav
        poor/*.wav

Works on any folder of .wav clips - built with TED-LIUM in mind (paired
with export_tedlium.py), but works on any generic audio+transcript folder.

IMPORTANT - about filler words specifically:
    TED-LIUM transcripts are typically CLEANED (um/uh mostly edited out),
    so filler_per_min from them would read artificially low. Because of
    that, this script does NOT compute filler_per_min at all by default -
    it relies on prepare_podcastfillers.py for that metric instead, which
    uses PodcastFillers' real annotated filler timestamps. Pass
    --attempt_filler_from_transcript if you want the (unreliable)
    transcript-keyword-matching estimate anyway.

Usage:
    python export_tedlium.py --n_samples 2000 --out_dir ./raw_tedlium   (see that script)
    python label_dataset.py --audio_dir ./raw_tedlium/clips \\
        --transcript_dir ./raw_tedlium/transcripts \\
        --output_dir ./data/labeled_clips --sample_for_review 5
"""

import argparse
import glob
import os

from preprocessing import preprocess_audio
from features import extract_pitch, detect_pauses, estimate_speech_rate, detect_filler_words
from rubric import label_from_features
from labeling_io import finalize


def compute_clip_features(audio_path: str, transcript_path: str | None,
                           attempt_filler_from_transcript: bool) -> dict:
    """Runs preprocessing + feature extraction for one clip."""
    audio, sr = preprocess_audio(audio_path)
    duration_sec = len(audio) / sr
    duration_min = max(duration_sec / 60.0, 1e-6)

    pitch = extract_pitch(audio, sr)
    pauses = detect_pauses(audio, sr)

    result = {
        "duration_sec": round(duration_sec, 1),
        "pitch_std_hz": round(pitch["pitch_std_hz"], 1),
        "pause_per_min": round(pauses["pause_count"] / duration_min, 2),
        "speech_rate_wpm": None,
        "filler_per_min": None,
    }

    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            words = f.read().strip().split()
        result["speech_rate_wpm"] = round(estimate_speech_rate(len(words), duration_sec), 1)

        if attempt_filler_from_transcript:
            fillers = detect_filler_words(words)
            result["filler_per_min"] = round(fillers["filler_count"] / duration_min, 2)

    return result


def find_transcript(transcript_dir: str | None, clip_id: str) -> str | None:
    if not transcript_dir:
        return None
    candidate = os.path.join(transcript_dir, clip_id + ".txt")
    return candidate if os.path.exists(candidate) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Folder of .wav clips to label")
    parser.add_argument("--transcript_dir", default=None,
                         help="Optional folder of matching .txt transcripts (same base filename)")
    parser.add_argument("--output_dir", required=True,
                         help="Where to write good/average/poor subfolders (matches train.py's --data_dir)")
    parser.add_argument("--report", default="tedlium_label_report.csv",
                         help="CSV file to write per-clip features + labels for review")
    parser.add_argument("--sample_for_review", type=int, default=0,
                         help="If >0, also copy N random clips per label into output_dir/_spot_check/ for manual listening")
    parser.add_argument("--dry_run", action="store_true",
                         help="Compute labels and write the CSV report only - don't copy any audio files yet")
    parser.add_argument("--attempt_filler_from_transcript", action="store_true",
                         help="Estimate fillers by keyword-matching the transcript - unreliable on cleaned TED-LIUM transcripts, off by default")
    args = parser.parse_args()

    clips = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not clips:
        raise ValueError(f"No .wav files found in {args.audio_dir}")

    print(f"Found {len(clips)} clips. Processing...")

    rows = []
    for i, clip_path in enumerate(clips, 1):
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        transcript_path = find_transcript(args.transcript_dir, clip_id)

        try:
            feats = compute_clip_features(clip_path, transcript_path, args.attempt_filler_from_transcript)
        except Exception as e:
            print(f"  [{i}/{len(clips)}] {clip_id}: FAILED ({e}) - skipping")
            continue

        label, scores = label_from_features(feats)
        rows.append({"clip_id": clip_id, "path": clip_path, "label": label, "source": "tedlium", **feats})
        print(f"  [{i}/{len(clips)}] {clip_id}: {label}  {feats}")

    finalize(rows, args.output_dir, args.report, args.dry_run, args.sample_for_review)


if __name__ == "__main__":
    main()
