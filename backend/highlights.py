from pathlib import Path
import json 
import re

MAX_VISUAL_DURATION = 40 
MIN_HIGHLIGHT_SCORE = 65

LAUGHTER_PATTERN = re .compile(
    r"\b(?:hahaha+|hehehe+|wkwk+|ngakak|ketawa|"
    r"laughing|laughter|lol|lmao)\b",
    re.IGNORECASE
)

SURPRISE_PATTERN = re.compile(
    r"\b(?:what the heck|what the hell|oh my god|"
    r"no way|are you serious|seriusan|astaga|"
    r"anjir|waduh|kok bisa|gak mungkin)\b",
    re.IGNORECASE
)

def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path.name}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def format_timestamp(seconds: float):
    total_seconds = int(max(0, seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) //60
    seconds_left = total_seconds % 60

    return(
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_left:02d}"
    )

def calculate_score(text: str):
    score = 0
    evidence = []

    if LAUGHTER_PATTERN.search(text):
        score += 78
        evidence.append(
            "Terdapat indikasi tawa dalam transkrip"
        )

    if SURPRISE_PATTERN.search(text):
        score += 68
        evidence.append(
            "Terdapat ungkapan terkejut atau reaksi spontan"
        )

    if text.count("!") >= 2:
        score += 15
        evidence.append(
            "Intonasi ekspresif terindikasi dari transkrip"
        )

    if text.count("?") >= 2:
        score += 10
        evidence.append(
            "Terdapat pertanyaan berulang atau ekspresif"
        )

    return min(score, 100), evidence


def detect_highlights(job_dir: Path, job_id: str):

    transcript = load_json(
        job_dir / "transcript.json"
    )

    visual = load_json(
        job_dir / "visual_analysis.json"
    )

    if transcript.get("status") != "success":
        raise ValueError(
            "Audio transcription is not successful"
        )

    if visual.get("status") != "success":
        raise ValueError(
            "Visual analysis is not successful"
        )

    if transcript.get("job_id") != job_id:
        raise ValueError(
            "Transcript job ID does not match"
        )

    if visual.get("job_id") != job_id:
        raise ValueError(
            "Visual analysis job ID does not match"
        )

    audio_start = float(
        transcript.get("start_seconds", 0)
    )

    visual_start = float(
        visual.get("start_seconds", 0)
    )

    if audio_start != visual_start:
        raise ValueError(
            "Audio and visual start times do not match"
        )

    visual_duration = float(
        visual.get(
            "analyzed_duration_seconds",
            MAX_VISUAL_DURATION
        )
    )

    analysis_duration = min(
        MAX_VISUAL_DURATION,
        visual_duration
    )

    segments = transcript.get("segments", [])

    if not isinstance(segments, list):
        raise ValueError(
            "Transcript segments must be a list"
        )

    visual_description = str(
        visual.get("description", "")
    ).strip()

    if not visual_description:
        raise ValueError(
            "Visual description is empty"
        )

    candidates = []

    for segment in segments:

        text = str(
            segment.get("text", "")
        ).strip()

        if not text:
            continue

        start = float(
            segment.get("start", 0)
        )

        end = float(
            segment.get("end", 0)
        )

        if end <= start:
            continue

        if start >= analysis_duration:
            continue

        end = min(
            end,
            analysis_duration
        )

        score, evidence = calculate_score(text)

        if score < MIN_HIGHLIGHT_SCORE:
            continue

        clip_start = max(
            0,
            start - 3
        )

        clip_end = min(
            analysis_duration,
            end + 4
        )

        vod_start = audio_start + clip_start
        vod_end = audio_start + clip_end

        candidate = {
            "type": "reaction_candidate",
            "score": score,
            "start": round(clip_start, 2),
            "end": round(clip_end, 2),
            "vod_start": round(vod_start, 2),
            "vod_end": round(vod_end, 2),
            "vod_timestamp": format_timestamp(
                vod_start
            ),
            "text": text,
            "reasons": evidence
        }

        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["start"]
        )
    )

    result = {
        "status": "success",
        "job_id": job_id,
        "analysis_type": "heuristic_v1",
        "analysis_duration_seconds": analysis_duration,
        "start_seconds": audio_start,
        "visual_context": visual_description,
        "total_candidates": len(candidates),
        "highlights": candidates,
        "limitations": [
            "Candidates are based on transcript cues",
            "Visual description is contextual only",
            "Scores are not probabilities of humor",
            "Transcript accuracy may affect results"
        ],
        "message": (
            "Highlight candidates generated"
            if candidates
            else "No strong highlight candidates detected"
        )
    }

    output_path = (
        job_dir / "funny_moments.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return result
