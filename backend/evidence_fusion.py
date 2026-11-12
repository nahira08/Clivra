
import argparse
import json
import re

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "long_vods"

VALID_JOB_ID = re.compile(
    r"vod_([0-9]+)_([0-9]+)_([0-9]+)"
)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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


def format_timestamp(seconds):
    total = max(0, int(float(seconds)))

    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def parse_timestamp(value):
    parts = str(value).split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid timestamp: {value}"
        )

    hours, minutes, seconds = [
        int(part) for part in parts
    ]

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def get_trigger_seconds(moment):
    trigger = moment.get(
        "trigger_timestamp"
    )

    if trigger:
        try:
            return parse_timestamp(trigger)
        except (ValueError, TypeError):
            pass

    value = moment.get("trigger_seconds")

    if value is not None:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

    return float(
        moment.get("start_seconds", 0)
    )


def get_twitch_url(video_id, seconds):
    total = max(0, int(seconds))

    hours = total // 3600
    minutes = (total % 3600) // 60
    remaining = total % 60

    time_value = (
        f"{hours}h{minutes}m{remaining}s"
    )

    return (
        f"https://www.twitch.tv/videos/"
        f"{quote(str(video_id), safe='')}"
        f"?t={time_value}"
    )


def index_by_id(items):
    result = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        item_id = item.get("id")

        if item_id is not None:
            result[str(item_id)] = item

    return result


def get_visual_summary(visual):
    if not visual:
        return {
            "available": False,
            "frames_analyzed": 0,
            "observations": [],
            "avatar_expressions": [],
            "limitations": []
        }

    observations = visual.get(
        "visual_observations",
        []
    )

    if not isinstance(observations, list):
        observations = []

    expressions = []
    limitations = []

    for observation in observations:
        expression = str(
            observation.get(
                "avatar_expression",
                ""
            )
        ).strip()

        limitation = str(
            observation.get(
                "limitations",
                ""
            )
        ).strip()

        if expression:
            expressions.append(expression)

        if limitation:
            limitations.append(limitation)

    return {
        "available": bool(observations),
        "frames_analyzed": visual.get(
            "frames_analyzed",
            0
        ),
        "observations": observations,
        "avatar_expressions": expressions,
        "limitations": limitations
    }


def get_audio_summary(audio):
    if not audio:
        return {
            "available": False,
            "classification": "unavailable",
            "energy_spike": False,
            "maximum_ratio": None,
            "spike_windows": [],
            "laughter_detected": None
        }

    analysis = audio.get(
        "analysis",
        {}
    )

    if not isinstance(analysis, dict):
        analysis = {}

    available = (
        analysis.get("status")
        == "completed"
    )

    return {
        "available": available,
        "classification": analysis.get(
            "classification",
            "unavailable"
        ),
        "energy_spike": (
            bool(
                analysis.get(
                    "audio_energy_spike",
                    False
                )
            )
            if available
            else False
        ),
        "maximum_ratio": analysis.get(
            "maximum_ratio"
        ),
        "spike_windows": analysis.get(
            "spike_windows",
            []
        ),
        "laughter_detected": None
    }


def get_transcript_context(
    transcript,
    start_seconds,
    end_seconds
):
    selected = []

    for segment in transcript.get(
        "segments",
        []
    ):
        if not isinstance(segment, dict):
            continue

        try:
            start = float(
                segment["vod_start"]
            )

            end = float(
                segment["vod_end"]
            )

            text = str(
                segment["text"]
            ).strip()

        except (
            KeyError,
            ValueError,
            TypeError
        ):
            continue

        if (
            text
            and end >= start_seconds
            and start <= end_seconds
        ):
            selected.append({
                "start_seconds": start,
                "end_seconds": end,
                "timestamp": format_timestamp(
                    start
                ),
                "text": text
            })

    selected.sort(
        key=lambda item: item[
            "start_seconds"
        ]
    )

    return selected


def decide_candidate(
    review_status,
    decision,
    evidence_verified,
    audio,
    visual
):
    if (
        review_status == "reviewed"
        and decision == "reject"
    ):
        return (
            "rejected",
            "AI rejected the transcript candidate. "
            "No sufficiently supported humor "
            "was identified in the dialogue."
        )

    if review_status != "reviewed":
        return (
            "needs_review",
            "The transcript AI review was not "
            "completed successfully."
        )

    if decision == "uncertain":
        return (
            "needs_review",
            "The transcript contains insufficient "
            "evidence for a confident decision."
        )

    if decision == "accept":
        if not evidence_verified:
            return (
                "needs_review",
                "The AI accepted the dialogue but "
                "the supporting quote was not "
                "verified against the transcript."
            )

        if not visual["available"]:
            return (
                "needs_review",
                "The dialogue may be humorous, "
                "but visual evidence is unavailable."
            )

        if not audio["available"]:
            return (
                "needs_review",
                "The dialogue may be humorous, "
                "but audio evidence is unavailable."
            )

        if audio["energy_spike"]:
            return (
                "needs_review",
                "The dialogue may contain humor "
                "and an audio energy increase "
                "was detected nearby. The audio "
                "source and actual reaction remain "
                "unverified."
            )

        return (
            "needs_review",
            "The dialogue may contain humor, "
            "but the visual and audio measurements "
            "do not independently confirm a "
            "humorous reaction."
        )

    return (
        "needs_review",
        "The AI decision could not be verified."
    )


def collect_evidence(
    moment,
    review_item,
    visual_item,
    audio_item,
    transcript,
    video_id
):
    review_status = (
        review_item.get(
            "review_status",
            "missing"
        )
        if review_item
        else "missing"
    )

    review = (
        review_item.get(
            "review",
            {}
        )
        if review_item
        else {}
    )

    if not isinstance(review, dict):
        review = {}

    decision = str(
        review.get(
            "decision",
            "uncertain"
        )
    ).lower()

    evidence_verified = bool(
        review.get(
            "evidence_verified",
            False
        )
    )

    visual = get_visual_summary(
        visual_item
    )

    audio = get_audio_summary(
        audio_item
    )

    trigger_seconds = get_trigger_seconds(
        moment
    )

    start_seconds = float(
        moment.get(
            "start_seconds",
            trigger_seconds
        )
    )

    end_seconds = float(
        moment.get(
            "end_seconds",
            trigger_seconds + 5
        )
    )

    if end_seconds < start_seconds:
        end_seconds = start_seconds + 5

    dialogue = get_transcript_context(
        transcript,
        max(0, start_seconds - 3),
        end_seconds + 3
    )

    status, reason = decide_candidate(
        review_status,
        decision,
        evidence_verified,
        audio,
        visual
    )

    evidence_notes = []

    if (
        decision == "accept"
        and evidence_verified
    ):
        evidence_notes.append(
            "Qwen identified potential humor "
            "and provided a transcript-grounded "
            "evidence quote."
        )

    if visual["available"]:
        evidence_notes.append(
            f"Qwen3-VL analyzed "
            f"{visual['frames_analyzed']} "
            f"static video frames."
        )

    if audio["available"]:
        if audio["energy_spike"]:
            evidence_notes.append(
                "An increase in relative audio "
                "energy was detected near the "
                "candidate timestamp."
            )
        else:
            evidence_notes.append(
                "Audio energy was measured, "
                "without a strong local spike."
            )

    limitations = [
        "Audio energy cannot distinguish "
        "laughter from speech, music, "
        "or game sound effects.",
        "Static images cannot reliably "
        "establish a genuine emotional "
        "reaction or laughter.",
        "A transcript-grounded AI explanation "
        "can still be inaccurate."
    ]

    return {
        "id": moment.get("id"),
        "status": status,
        "timestamp": format_timestamp(
            trigger_seconds
        ),
        "trigger_seconds": trigger_seconds,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "twitch_url": get_twitch_url(
            video_id,
            trigger_seconds
        ),
        "trigger_text": moment.get(
            "trigger_text",
            ""
        ),
        "heuristic_score": moment.get(
            "score"
        ),
        "transcript": dialogue,
        "transcript_review": {
            "review_status": review_status,
            "decision": decision,
            "reason": review.get(
                "reason",
                ""
            ),
            "evidence_quote": review.get(
                "evidence_quote",
                ""
            ),
            "evidence_verified": (
                evidence_verified
            ),
            "humor_type": review.get(
                "humor_type",
                "unverified"
            ),
            "error": review.get(
                "error"
            )
        },
        "visual_evidence": visual,
        "audio_evidence": audio,
        "evidence_notes": evidence_notes,
        "reason": reason,
        "limitations": limitations,
        "requires_manual_review": (
            status == "needs_review"
        )
    }


def fuse_job(job_id):
    match = VALID_JOB_ID.fullmatch(
        job_id
    )

    if not match:
        raise ValueError(
            "Invalid long VOD job ID."
        )

    job_dir = OUTPUT_DIR / job_id

    baseline = read_json(
        job_dir / "funny_moments.json"
    )

    reviews = read_json(
        job_dir / "humor_review.json"
    )

    visuals = read_json(
        job_dir / "visual_review.json"
    )

    audio = read_json(
        job_dir / "audio_analysis.json"
    )

    transcript = read_json(
        job_dir / "merged_transcript.json"
    )

    candidates = baseline.get(
        "moments",
        []
    )

    if not isinstance(candidates, list):
        raise ValueError(
            "Invalid candidate list."
        )

    review_index = index_by_id(
        reviews.get(
            "all_reviews",
            []
        )
    )

    visual_index = index_by_id(
        visuals.get(
            "moments",
            []
        )
    )

    audio_index = index_by_id(
        audio.get(
            "moments",
            []
        )
    )

    video_id = match.group(1)

    results = []

    for index, moment in enumerate(
        candidates,
        start=1
    ):
        moment_id = str(
            moment.get(
                "id",
                f"moment_{index:03d}"
            )
        )

        print(
            f"Fusing evidence "
            f"{index}/{len(candidates)}: "
            f"{moment_id}",
            flush=True
        )

        combined = collect_evidence(
            moment,
            review_index.get(
                moment_id
            ),
            visual_index.get(
                moment_id
            ),
            audio_index.get(
                moment_id
            ),
            transcript,
            video_id
        )

        results.append(combined)

        print(
            "  Final status:",
            combined["status"],
            flush=True
        )

    recommended = [
        item
        for item in results
        if item["status"] == "recommended"
    ]

    needs_review = [
        item
        for item in results
        if item["status"] == "needs_review"
    ]

    rejected = [
        item
        for item in results
        if item["status"] == "rejected"
    ]

    output = {
        "status": "completed",
        "job_id": job_id,
        "video_id": video_id,
        "method": "conservative_evidence_fusion_v1",
        "automatic_humor_confirmation": False,
        "laughter_classifier_available": False,
        "total_candidates": len(results),
        "recommended_count": len(recommended),
        "needs_review_count": len(needs_review),
        "rejected_count": len(rejected),
        "recommended_moments": recommended,
        "needs_review_moments": needs_review,
        "rejected_moments": rejected,
        "all_moments": results,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    output_path = (
        job_dir / "evidence_fusion.json"
    )

    save_json(
        output_path,
        output
    )

    return output, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clivra Evidence Fusion"
    )

    parser.add_argument(
        "job_id"
    )

    args = parser.parse_args()

    result, output_path = fuse_job(
        args.job_id
    )

    print()
    print("CLIVRA EVIDENCE FUSION")
    print("=" * 40)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Candidates:",
        result["total_candidates"]
    )

    print(
        "Recommended:",
        result["recommended_count"]
    )

    print(
        "Needs review:",
        result["needs_review_count"]
    )

    print(
        "Rejected:",
        result["rejected_count"]
    )

    print(
        "Saved to:",
        output_path
    )

    for item in result["all_moments"]:
        print()
        print(
            "Timestamp:",
            item["timestamp"]
        )

        print(
            "Final decision:",
            item["status"]
        )

        print(
            "Reason:",
            item["reason"]
        )

        print(
            "Twitch URL:",
            item["twitch_url"]
        )


if __name__ == "__main__":
    main()
