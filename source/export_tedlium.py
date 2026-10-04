"""
export_tedlium.py
------------------
Pulls a batch of utterance-level clips from TED-LIUM Release 3, saving
each one locally as a matched .wav + .txt pair, ready for label_dataset.py.

WHY THIS DOESN'T USE datasets.load_dataset():
    LIUM/tedlium still uses an old-style Python "loading script" on its
    main branch. Recent versions of the `datasets` library (4.0+) refuse
    to execute loading scripts at all, for security reasons - and it turns
    out this check happens before the library even looks at which revision
    you asked for, so even the "auto-converted Parquet mirror" workaround
    (revision="refs/convert/parquet") fails with the exact same error.

    So instead, this script uses `huggingface_hub` directly - a much
    lower-level library that just fetches files from a repo by name, with
    no script-execution logic to trip over. We list the Parquet files
    Hugging Face already auto-converted the dataset into, download them
    one at a time, and read rows out with pandas/pyarrow ourselves.

Sizing note (given no GPU, laptop-scale storage/RAM):
    A few thousand clips is a reasonable first-pass size - averaging
    ~10-15 seconds each, ~2,000 clips is roughly 6-8 hours of audio,
    typically a few GB on disk.

Usage:
    python export_tedlium.py --n_samples 20 --out_dir ../raw_tedlium_test   (try this first)
    python export_tedlium.py --n_samples 2000 --out_dir ../raw_tedlium
"""

import argparse
import io
import os

import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "LIUM/tedlium"
PARQUET_REVISION = "refs/convert/parquet"


def find_train_parquet_files(split: str) -> list:
    """
    Lists every file in the auto-converted Parquet mirror, filters down to
    the ones for the requested split, and prefers ones that look like
    release3 specifically if multiple releases are present as separate
    configs. Prints what it finds either way, since the exact folder
    naming for this repo hasn't been verified end-to-end before now.
    """
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=PARQUET_REVISION)

    print(f"  (found {len(all_files)} total files in the Parquet mirror)")

    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    split_files = [f for f in parquet_files if split.lower() in f.lower()]

    if not split_files:
        print(f"  No files matched split='{split}'. First 20 files found:")
        for f in all_files[:20]:
            print(f"    {f}")
        raise RuntimeError(
            f"Couldn't find any '{split}' parquet files - see the file list printed above "
            f"and adjust find_train_parquet_files() to match the actual naming."
        )

    release3_files = [f for f in split_files if "release3" in f.lower()]
    chosen = release3_files if release3_files else split_files

    print(f"  Matched {len(chosen)} parquet file(s) for split='{split}':")
    for f in chosen[:10]:
        print(f"    {f}")
    if len(chosen) > 10:
        print(f"    ... and {len(chosen) - 10} more")

    return chosen


def decode_audio_cell(audio_cell) -> tuple:
    """
    The 'audio' column in a HF Audio-feature parquet file is stored as a
    dict like {'bytes': b'...', 'path': 'some_original_name.wav'} - the
    raw encoded audio bytes, not a raw waveform array. Decode it here.
    """
    audio_bytes = audio_cell["bytes"]
    data, sr = sf.read(io.BytesIO(audio_bytes))
    return data, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000,
                         help="How many TED-LIUM utterance clips to export")
    parser.add_argument("--out_dir", type=str, default="./raw_tedlium",
                         help="Output folder - will contain clips/ and transcripts/ subfolders")
    parser.add_argument("--split", type=str, default="train",
                         help="Dataset split to pull from (train/validation/test)")
    parser.add_argument("--min_duration_sec", type=float, default=3.0,
                         help="Skip clips shorter than this - too short for meaningful rate/pitch/pause stats")
    args = parser.parse_args()

    clips_dir = os.path.join(args.out_dir, "clips")
    transcripts_dir = os.path.join(args.out_dir, "transcripts")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    print(f"Listing TED-LIUM Release 3 Parquet files ({args.split} split)...")
    parquet_files = find_train_parquet_files(args.split)

    exported = 0
    skipped_short = 0
    skipped_error = 0

    for pq_file in parquet_files:
        if exported >= args.n_samples:
            break

        print(f"\nDownloading {pq_file} (cached after first download)...")
        local_path = hf_hub_download(
            REPO_ID, pq_file, repo_type="dataset", revision=PARQUET_REVISION
        )

        print("Reading rows...")
        df = pd.read_parquet(local_path)

        for _, row in df.iterrows():
            if exported >= args.n_samples:
                break

            try:
                audio, sr = decode_audio_cell(row["audio"])
                text = str(row["text"]).strip()

                duration_sec = len(audio) / sr
                if duration_sec < args.min_duration_sec:
                    skipped_short += 1
                    continue

                if text.lower().startswith("ignore_time_segment"):
                    continue

                clip_id = f"tedlium_{exported:05d}"
                wav_path = os.path.join(clips_dir, clip_id + ".wav")
                txt_path = os.path.join(transcripts_dir, clip_id + ".txt")

                sf.write(wav_path, audio, sr)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                exported += 1
                if exported % 20 == 0:
                    print(f"  Exported {exported}/{args.n_samples}...")

            except Exception as e:
                skipped_error += 1
                continue

    print(f"\nDone. Exported {exported} clips to {clips_dir}/")
    print(f"Skipped: {skipped_short} too short, {skipped_error} errors")
    print(f"\nNext step:")
    print(f"  python label_dataset.py --audio_dir {clips_dir} --transcript_dir {transcripts_dir} \\")
    print(f"      --output_dir ../data/labeled_clips --sample_for_review 5")


if __name__ == "__main__":
    main()
