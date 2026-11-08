
from pathlib import Path
import gc
import json
import re
import wave

from faster_whisper import WhisperModel


MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

LONG_PAUSE = 0.85


def get_audio_duration(audio_path: Path):
    with wave.open(str(audio_path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def format_timestamp(seconds):
    total = int(max(0, seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remaining = total % 60

    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def clean_spacing(text):
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bUh -huh\b", "Uh-huh", text, flags=re.I)
    return text.strip()


def word_text(word):
    return re.sub(r"[^\w']", "", word["text"]).lower()


def has_ending(text):
    return bool(re.search(r"[.!?…][\"']?$", text.strip()))


def matches(words, index, pattern):
    if index + len(pattern) > len(words):
        return False

    return all(
        word_text(words[index + offset]) == expected
        for offset, expected in enumerate(pattern)
    )


def preceding_text(words, index, limit=12):
    start = max(0, index - limit)

    return " ".join(
        word_text(word)
        for word in words[start:index]
    ).strip()


def contextual_boundary(words, index):
    if index <= 0:
        return None

    previous = preceding_text(words, index)

    if matches(words, index, ["i", "think"]):
        if re.search(
            r"\bhow have you guys been$",
            previous
        ):
            return "?"

    if matches(words, index, ["i", "mean"]):
        if re.search(
            r"\b(?:dead|alive|right|okay|fine)$",
            previous
        ):
            return "."

    if matches(words, index, ["i", "hope"]):
        if re.search(r"\b(?:well|you|all)$", previous):
            return "."

    if matches(words, index, ["that's"]):
        if re.search(r"\b(?:alive|tired)$", previous):
            return "."

    return None


def question_ending(words):
    if not words:
        return False

    text = " ".join(word_text(word) for word in words)

    question_patterns = [
        r"^how (?:have|are|do|did|was|were)\b",
        r"^what (?:have|are|do|did|was|were)\b",
        r"^where (?:have|are|do|did|was|were)\b",
        r"^why (?:have|are|do|did|was|were)\b",
        r"^who (?:have|are|do|did|was|were)\b",
        r"^have you\b",
        r"^did you\b",
        r"^do you\b",
        r"^are you\b",
        r"^is it\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in question_patterns
    )


def join_words(words):
    parts = [word["text"] for word in words]
    return clean_spacing(" ".join(parts))


def create_phrase(words, pause_after, reason):
    if not words:
        return None

    text = join_words(words)

    if not text:
        return None

    start = words[0]["start"]
    end = words[-1]["end"]

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "text": text,
        "pause_after": round(pause_after, 2),
        "boundary_reason": reason,
        "words": [
            dict(word)
            for word in words
        ]
    }


def build_phrases(words):
    if not words:
        return []

    phrases = []
    current = []

    for index, word in enumerate(words):
        if current:
            punctuation = contextual_boundary(words, index)

            if punctuation:
                previous_word = current[-1]

                if not has_ending(previous_word["text"]):
                    previous_word["text"] += punctuation

                boundary = create_phrase(
                    current,
                    max(
                        0,
                        word["start"] - current[-1]["end"]
                    ),
                    "context"
                )

                if boundary:
                    phrases.append(boundary)

                current = []

        current.append(dict(word))

        is_last = index == len(words) - 1

        if is_last:
            phrase = create_phrase(
                current,
                0,
                "end_of_audio"
            )

            if phrase:
                phrases.append(phrase)

            break

        next_word = words[index + 1]

        pause = max(
            0,
            next_word["start"] - word["end"]
        )

        current_text = join_words(current)

        sentence_finished = has_ending(current_text)

        long_pause = pause >= LONG_PAUSE

        if sentence_finished or long_pause:
            reason = (
                "punctuation"
                if sentence_finished
                else "long_pause"
            )

            if (
                long_pause
                and not sentence_finished
                and question_ending(current)
            ):
                current[-1]["text"] += "?"
                reason = "question_and_pause"

            phrase = create_phrase(
                current,
                pause,
                reason
            )

            if phrase:
                phrases.append(phrase)

            current = []

    return phrases


def transcribe_with_pauses(
    job_dir: Path,
    job_id: str,
    start_seconds: int,
    language: str = "en"
):
    audio_path = job_dir / "audio.wav"

    if not audio_path.is_file():
        raise FileNotFoundError(
            "audio.wav not found"
        )

    audio_duration = get_audio_duration(audio_path)

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=4,
        num_workers=1
    )

    detected_language = language
    words = []

    try:
        segments_generator, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False
        )

        detected_language = info.language

        for segment in segments_generator:
            if not segment.words:
                continue

            for word in segment.words:
                text = word.word.strip()

                if not text:
                    continue

                start = max(0, float(word.start))
                end = min(
                    audio_duration,
                    float(word.end)
                )

                if end <= start:
                    continue

                words.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                    "probability": round(
                        float(word.probability),
                        3
                    )
                })

    finally:
        del model
        gc.collect()

    words.sort(
        key=lambda item: (
            item["start"],
            item["end"]
        )
    )

    for word in words:
        word["vod_start"] = round(
            start_seconds + word["start"],
            2
        )
        word["vod_end"] = round(
            start_seconds + word["end"],
            2
        )

    phrases = build_phrases(words)

    for phrase in phrases:
        phrase["vod_start"] = round(
            start_seconds + phrase["start"],
            2
        )
        phrase["vod_end"] = round(
            start_seconds + phrase["end"],
            2
        )
        phrase["vod_timestamp"] = format_timestamp(
            phrase["vod_start"]
        )

    pauses = []

    for index in range(len(words) - 1):
        gap = max(
            0,
            words[index + 1]["start"] - words[index]["end"]
        )

        if gap >= LONG_PAUSE:
            pauses.append({
                "start": words[index]["end"],
                "end": words[index + 1]["start"],
                "duration": round(gap, 2),
                "vod_start": round(
                    start_seconds + words[index]["end"],
                    2
                ),
                "vod_end": round(
                    start_seconds + words[index + 1]["start"],
                    2
                )
            })

    full_text = " ".join(
        phrase["text"]
        for phrase in phrases
    )

    result = {
        "status": "success",
        "job_id": job_id,
        "model": f"faster-whisper-{MODEL_SIZE}",
        "language": detected_language,
        "start_seconds": start_seconds,
        "audio_duration_seconds": round(
            audio_duration,
            2
        ),
        "long_pause_threshold": LONG_PAUSE,
        "total_words": len(words),
        "total_phrases": len(phrases),
        "total_long_pauses": len(pauses),
        "text": full_text,
        "phrases": phrases,
        "pauses": pauses,
        "message": "Pause-aware transcription completed"
    }

    output_path = job_dir / "pause_transcript.json"

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return result
