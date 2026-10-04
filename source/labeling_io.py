"""
labeling_io.py
---------------
Shared "write report -> sort into good/average/poor folders -> pull a
spot-check sample" logic, used by both label_dataset.py (TED-LIUM / any
generic audio+transcript folder) and prepare_podcastfillers.py.

Keeping this in one place means both sources produce output in exactly
the same shape, and end up in the same output_dir/good|average|poor/
structure that train.py expects.
"""

import csv
import os
import random
import shutil

REPORT_FIELDS = [
    "clip_id", "path", "label", "source", "duration_sec",
    "speech_rate_wpm", "filler_per_min", "pause_per_min", "pitch_std_hz",
]


def write_report(rows: list[dict], report_path: str) -> None:
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in REPORT_FIELDS})
    print(f"\nReport written to {report_path}")

    label_counts = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print(f"Label distribution: {label_counts}")


def sort_into_folders(rows: list[dict], output_dir: str, sample_for_review: int = 0) -> None:
    """
    Copies each row's audio file into output_dir/<label>/, and optionally
    pulls a random sample per label into output_dir/_spot_check/<label>/
    for manual listening.
    """
    for label in ("good", "average", "poor"):
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    for r in rows:
        dest = os.path.join(output_dir, r["label"], os.path.basename(r["path"]))
        shutil.copy2(r["path"], dest)

    print(f"Sorted {len(rows)} clips into {output_dir}/<good|average|poor>/")

    if sample_for_review > 0:
        spot_check_dir = os.path.join(output_dir, "_spot_check")
        for label in ("good", "average", "poor"):
            label_rows = [r for r in rows if r["label"] == label]
            sample = random.sample(label_rows, min(sample_for_review, len(label_rows)))
            label_dir = os.path.join(spot_check_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            for r in sample:
                shutil.copy2(r["path"], os.path.join(label_dir, os.path.basename(r["path"])))
        print(f"\nSpot-check sample copied to {spot_check_dir}/<good|average|poor>/ "
              f"- listen to these and compare against the label. If you disagree "
              f"often, adjust the ranges in rubric.py and re-run.")


def finalize(rows: list[dict], output_dir: str, report_path: str,
             dry_run: bool = False, sample_for_review: int = 0) -> None:
    """One call that does the whole write-report + sort-into-folders flow."""
    if not rows:
        print("No clips were successfully processed. Nothing to write.")
        return

    write_report(rows, report_path)

    if dry_run:
        print("\n--dry_run set - no audio files copied. Review the CSV, "
              "adjust rubric.py if needed, then re-run without --dry_run.")
        return

    sort_into_folders(rows, output_dir, sample_for_review)
