"""
merge_datasets.py
------------------
Combines the separate good/average/poor folders produced by
prepare_kaggle_tedlium.py and prepare_podcastfillers.py into a single
folder that train.py can point --data_dir at.

Each source script already writes its own output_dir/good|average|poor/
folders plus a CSV report (kaggle_tedlium_label_report.csv,
podcastfillers_label_report.csv). This script:
    1. Copies every clip from each source's good/average/poor into the
       combined output's matching good/average/poor folder.
    2. Concatenates the source CSV reports into one combined report, so
       you still have a single place to look up any clip's metrics/label
       and which dataset it came from (the "source" column already
       distinguishes them: "tedlium1_kaggle" vs "podcastfillers").

Filename collisions between the two sources are extremely unlikely
(TED-LIUM clips are named like "AaronHuey_2010X_win00.wav", PodcastFillers
clips like "pf_001_0000.wav"), but this script checks and warns rather
than silently overwriting, just in case.

This is meant to be run ONCE, after you're happy with both sources'
labeled output (i.e. after the hybrid spot-check step) - re-running it
will re-copy everything and duplicate rows in the combined report if the
same source reports are passed in twice, so don't run it repeatedly as
part of a loop.

Usage:
    python merge_datasets.py \\
        --sources ../data/labeled_clips ../data/labeled_clips_pf \\
        --reports kaggle_tedlium_label_report.csv podcastfillers_label_report.csv \\
        --output_dir ../data/labeled_clips_combined \\
        --combined_report combined_label_report.csv
"""

import argparse
import csv
import os
import shutil

from labeling_io import REPORT_FIELDS

LABELS = ("good", "average", "poor")


def copy_source_clips(source_dir: str, output_dir: str) -> dict:
    """Copies one source's good/average/poor clips into the combined
    output folder. Returns a dict of per-label copy counts."""
    counts = {label: 0 for label in LABELS}
    for label in LABELS:
        src_label_dir = os.path.join(source_dir, label)
        if not os.path.isdir(src_label_dir):
            print(f"  (no '{label}' folder found under {source_dir} - skipping)")
            continue

        dest_label_dir = os.path.join(output_dir, label)
        os.makedirs(dest_label_dir, exist_ok=True)

        for fname in os.listdir(src_label_dir):
            src_path = os.path.join(src_label_dir, fname)
            if not os.path.isfile(src_path):
                continue
            dest_path = os.path.join(dest_label_dir, fname)
            if os.path.exists(dest_path):
                print(f"  WARNING: filename collision on '{fname}' in '{label}' - "
                      f"keeping the first copy, skipping this one. Check your two "
                      f"sources aren't using overlapping clip_id naming.")
                continue
            shutil.copy2(src_path, dest_path)
            counts[label] += 1

    return counts


def load_report_rows(report_path: str) -> list:
    if not report_path or not os.path.exists(report_path):
        print(f"  (no report found at {report_path} - combined report will be missing these rows)")
        return []
    with open(report_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True,
                         help="Each source's output_dir (the folder containing good/average/poor)")
    parser.add_argument("--reports", nargs="*", default=[],
                         help="Each source's CSV report, in the SAME ORDER as --sources. "
                              "Optional per-source - omit an entry (or the whole flag) if you "
                              "don't need the combined report, but then the row won't appear in it.")
    parser.add_argument("--output_dir", required=True,
                         help="Where the combined good/average/poor folders go")
    parser.add_argument("--combined_report", default="combined_label_report.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_counts = {label: 0 for label in LABELS}
    all_rows = []

    for i, source_dir in enumerate(args.sources):
        print(f"\nMerging source {i + 1}/{len(args.sources)}: {source_dir}")
        counts = copy_source_clips(source_dir, args.output_dir)
        for label in LABELS:
            print(f"  {label}: {counts[label]} clips copied")
            total_counts[label] += counts[label]

        report_path = args.reports[i] if i < len(args.reports) else None
        rows = load_report_rows(report_path)
        all_rows.extend(rows)

    if all_rows:
        with open(args.combined_report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k) for k in REPORT_FIELDS})
        print(f"\nCombined report written to {args.combined_report} ({len(all_rows)} rows)")

    print(f"\nTotal merged into {args.output_dir}/<good|average|poor>/:")
    for label in LABELS:
        print(f"  {label}: {total_counts[label]}")
    print(f"  TOTAL: {sum(total_counts.values())}")


if __name__ == "__main__":
    main()
