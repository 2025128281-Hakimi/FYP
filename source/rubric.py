INF = float("inf")

RUBRIC_RANGES = {
    "speech_rate_wpm": {
        "good": [(120, 160)],
        "average": [(110, 120)],
    },
    "filler_per_min": {
        "good": [(0, 5)],
        "average": [(5, 12)],
    },
    "pause_per_min": {
        "good": [(4, 8)],
        "average": [(3, 4), (8, 15)],
    },
    "pitch_std_hz": {
        "good": [(20, 40)],
        "average": [(10, 20), (40, INF)],
    },
}


def _in_any(value: float, ranges: list) -> bool:
    return any(lo <= value <= hi for lo, hi in ranges)


def _classify_value(value: float, ranges: dict) -> str:
    if _in_any(value, ranges["good"]):
        return "good"
    if _in_any(value, ranges["average"]):
        return "average"
    return "poor"


def score_metrics(features: dict) -> dict:
    scores = {}
    for metric_name, ranges in RUBRIC_RANGES.items():
        if metric_name in features and features[metric_name] is not None:
            scores[metric_name] = _classify_value(features[metric_name], ranges)
    return scores


def combine_label(scores: dict) -> str:
    if not scores:
        return "average"

    good_count = sum(1 for v in scores.values() if v == "good")
    poor_count = sum(1 for v in scores.values() if v == "poor")

    if good_count >= 3:
        return "good"
    if poor_count >= 2:
        return "poor"
    return "average"


def label_from_features(features: dict) -> tuple[str, dict]:
    scores = score_metrics(features)
    return combine_label(scores), scores


if __name__ == "__main__":
    examples = [
        {"speech_rate_wpm": 135, "filler_per_min": 2, "pause_per_min": 6, "pitch_std_hz": 30},
        {"speech_rate_wpm": 115, "filler_per_min": 8, "pause_per_min": 12, "pitch_std_hz": 17},
        {"speech_rate_wpm": 100, "filler_per_min": 14, "pause_per_min": 2, "pitch_std_hz": 8},
    ]
    for ex in examples:
        label, scores = label_from_features(ex)
        print(f"{ex} -> {label}  (per-metric: {scores})")
