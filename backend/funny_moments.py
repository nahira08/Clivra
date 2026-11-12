
import argparse
import json
import re

from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "long_vods"

MAX_RESULTS = 10
MIN_SCORE = 35
CONTEXT_BEFORE = 10.0
CONTEXT_AFTER = 12.0
MAX_MOMENT_DURATION = 45.0


CUES = [
    {
        "name": "laughter",
        "weight": 40,
        "reason": "Terdapat indikasi tawa dalam percakapan.",
        "patterns": [
            r"\b(?:hahaha+|hehehe+|lol|lmao|rofl)\b",
            r"\b(?:i'm laughing|can't stop laughing)\b",
            r"\b(?:laughs|laughing|giggles|chuckles)\b",
        ],
    },
    {
        "name": "contradiction",
        "weight": 35,
        "reason": "Terdapat pernyataan yang berlawanan atau tidak terduga.",
        "patterns": [
            r"\b(?:but wait|wait no|actually no)\b",
            r"\b(?:that makes no sense|doesn't make sense)\b",
            r"\b(?:i don't even know|i can't remember)\b",
            r"\b(?:you can't be|you cannot be)\b",
            r"\b(?:what do you mean)\b",
        ],
    },
    {
        "name": "unexpected_reaction",
        "weight": 30,
        "reason": "Terdapat reaksi spontan yang berpotensi menarik.",
        "patterns": [
            r"\b(?:what the heck|what the hell|what the fuck)\b",
            r"\b(?:no way|oh my god|oh my gosh)\b",
            r"\b(?:are you serious|are you kidding)\b",
            r"\b(?:i can't believe|what just happened)\b",
            r"\b(?:excuse me|hold on a second)\b",
        ],
    },
    {
        "name": "self_correction",
        "weight": 25,
        "reason": "Pembicara mengoreksi atau membalikkan ucapannya.",
        "patterns": [
            r"\b(?:i mean|never mind|nevermind)\b",
            r"\b(?:no no no|wait wait wait)\b",
            r"\b(?:that's not what i meant)\b",
            r"\b(?:forget what i said)\b",
        ],
    },
    {
        "name": "playful_exchange",
        "weight": 20,
        "reason": "Terdapat ungkapan yang mungkin merupakan candaan.",
        "patterns": [
            r"\b(?:i'm joking|just kidding|it's a joke)\b",
            r"\b(?:you're kidding|stop trolling)\b",
            r"\b(?:skill issue|you're trolling)\b",
        ],
    },
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def format_timestamp(seconds):
    total = max(0, int(seconds))

    hours = total // 3600
    minutes = (total % 3600) // 60
    remaining = total % 60

    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temporary.replace(path)


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()


def validate_segments(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid transcript document")

    if not isinstance(data.get("segments"), list):
        raise ValueError("Transcript segments are missing")

    validated = []

    for segment in data["segments"]:
        if not isinstance(segment, dict):
            continue

        text = segment.get("text")
        start = segment.get("vod_start")
        end = segment.get("vod_end")

        if not isinstance(text, str):
            continue

        if not isinstance(start, (int, float)):
            continue

        if not isinstance(end, (int, float)):
            continue

        if start < 0 or end <= start:
            continue

        if not text.strip():
            continue

        validated.append({
            "text": text.strip(),
            "vod_start": float(start),
            "vod_end": float(end),
        })

    validated.sort(
        key=lambda segment: segment["vod_start"]
    )

    return validated


def analyze_dialogue(text):
    normalized = normalize_text(text)

    evidence = []
    score = 0

    for cue in CUES:
        matched = any(
            re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE
            )
            for pattern in cue["patterns"]
        )

        if matched:
            score += cue["weight"]

            evidence.append({
                "type": cue["name"],
                "reason": cue["reason"],
            })

    if len(evidence) >= 2:
        score += 10

    return {
        "score": min(score, 100),
        "evidence": evidence,
    }


def get_context(segments, start, end):
    selected = [
        segment
        for segment in segments
        if (
            segment["vod_end"] >= start
            and segment["vod_start"] <= end
        )
    ]

    return selected


def build_candidate(segments, index):
    anchor = segments[index]

    analysis = analyze_dialogue(
        anchor["text"]
    )

    if analysis["score"] < MIN_SCORE:
        return None

    start = max(
        0.0,
        anchor["vod_start"] - CONTEXT_BEFORE
    )

    end = (
        anchor["vod_end"] + CONTEXT_AFTER
    )

    if end - start > MAX_MOMENT_DURATION:
        end = start + MAX_MOMENT_DURATION

    context = get_context(
        segments,
        start,
        end
    )

    if not context:
        return None

    actual_start = max(
        start,
        context[0]["vod_start"]
    )

    actual_end = min(
        end,
        context[-1]["vod_end"]
    )

    dialogue = " ".join(
        segment["text"]
        for segment in context
    )

    reasons = [
        item["reason"]
        for item in analysis["evidence"]
    ]

    return {
        "start_seconds": round(actual_start, 2),
        "end_seconds": round(actual_end, 2),
        "timestamp": format_timestamp(actual_start),
        "duration_seconds": round(
            actual_end - actual_start,
            2
        ),
        "trigger_timestamp": format_timestamp(
            anchor["vod_start"]
        ),
        "trigger_text": anchor["text"],
        "dialogue": dialogue,
        "score": analysis["score"],
        "evidence": analysis["evidence"],
        "reason": " ".join(reasons),
        "needs_review": True,
    }


def overlap_ratio(first, second):
    overlap = max(
        0.0,
        min(
            first["end_seconds"],
            second["end_seconds"]
        ) - max(
            first["start_seconds"],
            second["start_seconds"]
        )
    )

    first_duration = max(
        0.01,
        first["end_seconds"] - first["start_seconds"]
    )

    second_duration = max(
        0.01,
        second["end_seconds"] - second["start_seconds"]
    )

    return overlap / min(
        first_duration,
        second_duration
    )


def remove_duplicates(candidates):
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            item["start_seconds"]
        )
    )

    selected = []

    for candidate in ranked:
        duplicate = any(
            overlap_ratio(candidate, existing) >= 0.5
            for existing in selected
        )

        if not duplicate:
            selected.append(candidate)

        if len(selected) >= MAX_RESULTS:
            break

    return selected


def detect_funny_moments(transcript):
    segments = validate_segments(
        transcript
    )

    candidates = []

    for index in range(len(segments)):
        candidate = build_candidate(
            segments,
            index
        )

        if candidate is not None:
            candidates.append(candidate)

    selected = remove_duplicates(
        candidates
    )

    for index, candidate in enumerate(selected, start=1):
        candidate["id"] = f"moment_{index:03d}"

    return {
        "status": "completed",
        "video_id": transcript.get("video_id"),
        "title": transcript.get("title"),
        "vod_duration": transcript.get("vod_duration"),
        "detector": "clivra-heuristic-baseline-v1",
        "is_ai_verified": False,
        "requires_human_review": True,
        "analyzed_speech_segments": len(segments),
        "total_candidates": len(selected),
        "moments": selected,
        "generated_at": utc_now(),
    }


def run_detection(job_id):
    if not re.fullmatch(
        r"vod_[0-9]+_[0-9]+_[0-9]+",
        job_id
    ):
        raise ValueError("Invalid long VOD job ID")

    job_dir = OUTPUT_DIR / job_id

    transcript_path = (
        job_dir / "merged_transcript.json"
    )

    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"Transcript not found: {transcript_path}"
        )

    transcript = load_json(
        transcript_path
    )

    result = detect_funny_moments(
        transcript
    )

    output_path = (
        job_dir / "funny_moments.json"
    )

    save_json(
        output_path,
        result
    )

    return result, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clivra Funny Moment Baseline"
    )

    parser.add_argument(
        "job_id",
        help="Existing long VOD job ID"
    )

    args = parser.parse_args()

    result, output_path = run_detection(
        args.job_id
    )

    print()
    print("CLIVRA FUNNY MOMENT DETECTION")
    print("=" * 45)

    print(
        "Speech segments analyzed:",
        result["analyzed_speech_segments"]
    )

    print(
        "Candidates found:",
        result["total_candidates"]
    )

    print(
        "Output:",
        output_path
    )

    if not result["moments"]:
        print()
        print(
            "No sufficiently strong humor cues "
            "found in this transcript."
        )

    for moment in result["moments"]:
        print()
        print("-" * 45)

        print(
            "Timestamp:",
            moment["timestamp"]
        )

        print(
            "Score:",
            moment["score"]
        )

        print(
            "Trigger:",
            moment["trigger_text"]
        )

        print(
            "Reason:",
            moment["reason"]
        )

    print()
    print(
        "Note: Results are heuristic candidates, "
        "not AI-verified funny moments."
    )


if __name__ == "__main__":
    main()
