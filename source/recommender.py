import json
import os

from rubric import RUBRIC_RANGES

DEFAULT_MODEL = "gpt-4o-mini"
MAX_ITEMS = 3

# Only these fields are sent to OpenAI - never the audio or anything else from the browser.
METRIC_FIELDS = {
    "wpm": "speech rate (words per minute)",
    "fillerCount": "filler words (uh/um) counted",
    "pauseCount": "noticeable pauses counted",
    "pitchStd": "pitch variation (standard deviation, Hz)",
    "durationSec": "presentation length (seconds)",
    "qualityLabel": "overall quality from the Conformer model (good/average/poor)",
    "overallScore": "overall score out of 100",
}

SYSTEM_PROMPT = """You are a friendly public-speaking coach inside a presentation coaching app.
You receive measurements taken from one practice presentation and write short, personalised
feedback for the speaker.

Reference ranges used by the app (per minute values unless stated):
{ranges}

Rules:
- Base every point only on the measurements given. Do not invent problems or numbers.
- A measurement that is null was not available; do not mention it.
- Mention the speaker's actual numbers where helpful, and convert counts to per-minute
  rates using the presentation length.
- Give practical, specific actions the speaker can try next time.
- Write {max_items} or fewer strengths and {max_items} or fewer improvements.
- Each title is at most 6 words. Each text is 1-2 sentences, plain English, no markdown.

Reply with JSON only, in exactly this shape:
{{"overall": "2-3 sentence summary",
  "strengths": [{{"title": "...", "text": "..."}}],
  "improvements": [{{"title": "...", "text": "..."}}]}}"""


class RecommenderUnavailable(Exception):
    pass


def _ranges_text() -> str:
    lines = []
    for metric, levels in RUBRIC_RANGES.items():
        parts = [f"{level} {', '.join(f'{lo:g}-{hi:g}' for lo, hi in spans)}"
                 for level, spans in levels.items()]
        lines.append(f"- {metric}: " + "; ".join(parts) + "; anything else is poor")
    return "\n".join(lines).replace("-inf", " and above")


def _clean_items(items) -> list:
    cleaned = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("title") and item.get("text"):
            cleaned.append({"title": str(item["title"])[:80], "text": str(item["text"])[:400]})
    return cleaned[:MAX_ITEMS]


def generate_recommendations(results: dict) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RecommenderUnavailable("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError:
        raise RecommenderUnavailable("the openai package is not installed")

    metrics = {desc: results.get(key) for key, desc in METRIC_FIELDS.items()}
    client = OpenAI(timeout=20, max_retries=1)
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": SYSTEM_PROMPT.format(ranges=_ranges_text(), max_items=MAX_ITEMS)},
            {"role": "user", "content": "Measurements:\n" + json.dumps(metrics, indent=2)},
        ],
    )

    data = json.loads(response.choices[0].message.content)
    result = {
        "overall": str(data.get("overall", ""))[:600],
        "strengths": _clean_items(data.get("strengths")),
        "improvements": _clean_items(data.get("improvements")),
    }
    if not result["strengths"] and not result["improvements"]:
        raise ValueError("the AI reply had no recommendations")
    return result
