

RUBRIC_RANGES = {
    "speech_rate_wpm": {
        "good": (110, 160),
        "average": (90, 180),   # anything in here but outside "good" = average
    },
    "filler_per_min": {
        "good": (0, 2),
        "average": (0, 5),
    },
    "pause_per_min": {
    "good": (0, 15),
    "average": (0, 25),
},
    "pitch_std_hz": {
        "good": (15, 70),
        "average": (5, 100),
    },
}


def _classify_value(value: float, ranges: dict) -> str:
    """Classify a single metric value as good / average / poor."""
    good_lo, good_hi = ranges["good"]
    avg_lo, avg_hi = ranges["average"]

    if good_lo <= value <= good_hi:
        return "good"
    if avg_lo <= value <= avg_hi:
        return "average"
    return "poor"


def score_metrics(features: dict) -> dict:
    """
    Takes a dict that may contain any of:
        speech_rate_wpm, filler_per_min, pause_per_min, pitch_std_hz
    (missing keys are simply skipped)

    Returns a dict of the same keys mapped to "good" / "average" / "poor".
    """
    scores = {}
    for metric_name, ranges in RUBRIC_RANGES.items():
        if metric_name in features and features[metric_name] is not None:
            scores[metric_name] = _classify_value(features[metric_name], ranges)
    return scores


def combine_label(scores: dict) -> str:
    """
    Combines per-metric good/average/poor scores into one overall label.
    See module docstring for the combination rule.
    """
    if not scores:
        return "average"  # no data at all - safest fallback, should be rare

    good_count = sum(1 for v in scores.values() if v == "good")
    poor_count = sum(1 for v in scores.values() if v == "poor")

    if good_count >= 3:
        return "good"
    if poor_count >= 2:
        return "poor"
    return "average"


def label_from_features(features: dict) -> tuple[str, dict]:
    """Convenience wrapper: features in, (overall_label, per_metric_scores) out."""
    scores = score_metrics(features)
    return combine_label(scores), scores


if __name__ == "__main__":
    # Quick self-check with a few made-up examples - no audio needed.
    examples = [
        {"speech_rate_wpm": 135, "filler_per_min": 1, "pause_per_min": 2, "pitch_std_hz": 40},
        {"speech_rate_wpm": 210, "filler_per_min": 9, "pause_per_min": 10, "pitch_std_hz": 3},
        {"speech_rate_wpm": 100, "filler_per_min": 4, "pause_per_min": 5, "pitch_std_hz": 20},
    ]
    for ex in examples:
        label, scores = label_from_features(ex)
        print(f"{ex} -> {label}  (per-metric: {scores})")
