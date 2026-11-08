
from pathlib import Path
from threading import Thread, Lock
from datetime import datetime, timezone
import json
import re
import shutil
import subprocess


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"

CHUNK_SIZE = 20
MAX_OUTPUT_TOKENS = 384
CONTEXT_SIZE = 4096
MODEL_TIMEOUT = 300

LONG_PAUSE = 0.85
MEDIUM_PAUSE = 0.65
MAX_SENTENCE_DURATION = 12.0

VALID_MARKS = {",", ".", "?", "!", "..."}

refinement_lock = Lock()
active_job = None


def save_json(path, data):
    path = Path(path)
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


def set_status(
    job_dir,
    job_id,
    status,
    message,
    **extra
):
    data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        **extra
    }

    save_json(
        job_dir / "refinement_status.json",
        data
    )

    return data


def split_word(text):
    text = str(text).strip()

    match = re.search(
        r"(\.\.\.|[,.!?])$",
        text
    )

    mark = match.group(1) if match else None

    if match:
        text = text[:match.start()].strip()

    return text, mark


def load_words(transcript):
    words = []
    phrase_boundaries = set()

    for phrase in transcript.get("phrases", []):
        phrase_words = phrase.get("words", [])

        for item in phrase_words:
            text, mark = split_word(
                item.get("text", "")
            )

            if not text:
                continue

            words.append({
                "start": float(item["start"]),
                "end": float(item["end"]),
                "vod_start": float(item["vod_start"]),
                "vod_end": float(item["vod_end"]),
                "text": text,
                "original_mark": mark,
                "probability": item.get("probability")
            })

        if phrase_words and words:
            phrase_boundaries.add(len(words))

    if not words:
        raise ValueError(
            "No words found in pause_transcript.json"
        )

    return words, phrase_boundaries


def create_chunks(words):
    chunks = []

    for start in range(
        0,
        len(words),
        CHUNK_SIZE
    ):
        end = min(
            start + CHUNK_SIZE,
            len(words)
        )

        chunks.append({
            "start": start,
            "end": end,
            "words": words[start:end]
        })

    return chunks


def build_prompt(chunk):
    lines = []
    words = chunk["words"]

    for index, word in enumerate(words):
        position = index + 1

        if index + 1 < len(words):
            next_word = words[index + 1]

            pause = max(
                0,
                next_word["start"] - word["end"]
            )
        else:
            pause = 0

        original_mark = (
            word["original_mark"] or "-"
        )

        lines.append(
            f'{position}:{word["text"]}'
            f'|pause={pause:.2f}'
            f'|punct={original_mark}'
        )

    return (
        "You restore punctuation in an English "
        "livestream transcript.\n"
        "Words are numbered from 1.\n"
        "Do not change or reorder words.\n"
        "Use grammar, meaning, and pause duration.\n"
        "Keep good existing punctuation.\n"
        "A pause does not always end a sentence.\n"
        "Return ONLY punctuation changes.\n"
        "Do NOT output an entry for words that "
        "need no punctuation.\n"
        "Do NOT output spaces as punctuation.\n"
        "Allowed marks: , . ? ! ...\n"
        "Output compact JSON only.\n"
        'Format: {"marks":[{"after":5,"mark":"?"}]}\n'
        "If no changes are needed, output "
        '{"marks":[]}\n'
        "No explanation. No markdown.\n\n"
        "WORDS:\n"
        + "\n".join(lines)
        + "\n\nANSWER:\n"
    )


def extract_json(output):
    cleaned = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        output
    )

    cleaned = re.split(
        r"\[\s*Prompt:.*?Generation:.*?\]",
        cleaned,
        maxsplit=1,
        flags=re.DOTALL
    )[0]

    cleaned = cleaned.split(
        "Exiting...",
        1
    )[0]

    if "\n> " in cleaned:
        cleaned = cleaned.rsplit(
            "\n> ",
            1
        )[-1]

    decoder = json.JSONDecoder()
    candidates = []

    for match in re.finditer(r"[\{\[]", cleaned):
        try:
            parsed, _ = decoder.raw_decode(
                cleaned[match.start():]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            if isinstance(
                parsed.get("marks"),
                list
            ):
                candidates.append(parsed["marks"])

        elif isinstance(parsed, list):
            if all(
                isinstance(item, dict)
                and "after" in item
                and "mark" in item
                for item in parsed
            ):
                candidates.append(parsed)

    if not candidates:
        raise ValueError(
            "Qwen did not return valid JSON"
        )

    return candidates[-1]


def validate_marks(raw_marks, chunk_size):
    marks = {}

    if not isinstance(raw_marks, list):
        raise ValueError(
            "Punctuation response must be a list"
        )

    for item in raw_marks:
        if not isinstance(item, dict):
            raise ValueError(
                "Invalid punctuation item"
            )

        position = item.get("after")
        mark = item.get("mark")

        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 1
            or position > chunk_size
        ):
            raise ValueError(
                "Invalid punctuation position"
            )

        if mark is None:
            continue

        if not isinstance(mark, str):
            raise ValueError(
                "Invalid punctuation value"
            )

        mark = mark.strip()

        if mark == "":
            continue

        if mark not in VALID_MARKS:
            raise ValueError(
                f"Unsupported punctuation: {mark}"
            )

        if position in marks:
            raise ValueError(
                "Duplicate punctuation position"
            )

        marks[position] = mark

    return marks


def analyze_chunk(
    llama,
    chunk,
    job_dir,
    chunk_number
):
    prompt = build_prompt(chunk)

    command = [
        llama,
        "-hf",
        MODEL_NAME,
        "-ngl",
        "0",
        "--no-mmproj-offload",
        "-c",
        str(CONTEXT_SIZE),
        "-n",
        str(MAX_OUTPUT_TOKENS),
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "-p",
        prompt
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=MODEL_TIMEOUT
        )

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Qwen chunk {chunk_number} timed out"
        ) from error

    except OSError as error:
        raise RuntimeError(
            f"Cannot run llama-cli: {error}"
        ) from error

    raw_filename = (
        f"refinement_chunk_{chunk_number}_raw.txt"
    )

    raw_path = job_dir / raw_filename

    raw_path.write_text(
        "========== STDOUT ==========\n"
        + result.stdout
        + "\n========== STDERR ==========\n"
        + result.stderr,
        encoding="utf-8"
    )

    if result.returncode != 0:
        details = (
            result.stderr or result.stdout
        ).strip()

        raise RuntimeError(
            f"Qwen failed: {details[-700:]}"
        )

    raw_marks = extract_json(
        result.stdout
    )

    marks = validate_marks(
        raw_marks,
        len(chunk["words"])
    )

    return marks


def build_base_marks(words):
    return {
        index + 1: word["original_mark"]
        for index, word in enumerate(words)
        if word["original_mark"]
    }


def apply_ai_marks(
    base_marks,
    chunks,
    chunk_results
):
    marks = dict(base_marks)

    for chunk, suggested_marks in zip(
        chunks,
        chunk_results
    ):
        for local_position, mark in suggested_marks.items():
            global_position = (
                chunk["start"] + local_position
            )

            original = base_marks.get(
                global_position
            )

            if original is None:
                marks[global_position] = mark

            elif (
                original == "."
                and mark == "?"
            ):
                marks[global_position] = "?"

    return marks


def ensure_boundaries(
    words,
    marks,
    phrase_boundaries
):
    result = dict(marks)

    for index in range(len(words) - 1):
        position = index + 1

        current = words[index]
        following = words[index + 1]

        gap = max(
            0,
            following["start"] - current["end"]
        )

        existing = result.get(position)

        if existing in {".", "?", "!", "..."}:
            continue

        if gap >= LONG_PAUSE:
            result[position] = "."

        elif (
            gap >= MEDIUM_PAUSE
            and position in phrase_boundaries
            and following["text"][:1].isupper()
        ):
            result[position] = "."

    return result


def ensure_max_duration(
    words,
    marks,
    phrase_boundaries
):
    result = dict(marks)
    sentence_start = 0

    for index, word in enumerate(words):
        position = index + 1

        elapsed = (
            word["end"]
            - words[sentence_start]["start"]
        )

        mark = result.get(position)

        if elapsed > MAX_SENTENCE_DURATION:
            candidates = [
                boundary
                for boundary in phrase_boundaries
                if sentence_start < boundary <= position
            ]

            split_at = (
                candidates[-1]
                if candidates
                else position
            )

            if result.get(split_at) not in {
                ".", "?", "!", "..."
            }:
                result[split_at] = "."

            sentence_start = split_at

        elif mark in {".", "?", "!", "..."}:
            sentence_start = position

    return result


def make_sentence(
    words,
    marks,
    start,
    end
):
    selected = words[start:end]
    tokens = []

    for index in range(start, end):
        word = words[index]
        text = word["text"]

        if index == start and text:
            text = (
                text[0].upper()
                + text[1:]
            )

        mark = marks.get(
            index + 1
        )

        if mark:
            text += mark

        tokens.append(text)

    text = " ".join(tokens)

    text = re.sub(
        r"\s+([,.!?])",
        r"\1",
        text
    )

    text = re.sub(
        r"\s+-",
        "-",
        text
    )

    return {
        "start": round(
            selected[0]["start"],
            2
        ),
        "end": round(
            selected[-1]["end"],
            2
        ),
        "vod_start": round(
            selected[0]["vod_start"],
            2
        ),
        "vod_end": round(
            selected[-1]["vod_end"],
            2
        ),
        "text": text,
        "word_start_index": start + 1,
        "word_end_index": end
    }


def assemble_sentences(words, marks):
    sentences = []
    start = 0

    for index in range(len(words)):
        position = index + 1

        mark = marks.get(
            position
        )

        if mark in {".", "?", "!", "..."}:
            sentences.append(
                make_sentence(
                    words,
                    marks,
                    start,
                    index + 1
                )
            )

            start = index + 1

    if start < len(words):
        sentences.append(
            make_sentence(
                words,
                marks,
                start,
                len(words)
            )
        )

    return sentences


def validate_sentences(words, sentences):
    if not sentences:
        raise ValueError(
            "No sentences generated"
        )

    expected_start = 1

    for sentence in sentences:
        if (
            sentence["word_start_index"]
            != expected_start
        ):
            raise ValueError(
                "Sentence word indices are not continuous"
            )

        if (
            sentence["word_end_index"]
            < expected_start
        ):
            raise ValueError(
                "Invalid sentence word range"
            )

        expected_start = (
            sentence["word_end_index"]
            + 1
        )

    if expected_start != len(words) + 1:
        raise ValueError(
            "Some words were omitted"
        )


def process_refinement(job_dir, job_id):
    global active_job

    try:
        set_status(
            job_dir,
            job_id,
            "processing",
            "Reading existing transcript"
        )

        source = (
            job_dir / "pause_transcript.json"
        )

        if not source.is_file():
            raise FileNotFoundError(
                "pause_transcript.json not found"
            )

        transcript = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )

        if transcript.get("job_id") != job_id:
            raise ValueError(
                "Job ID does not match"
            )

        if transcript.get("status") != "success":
            raise ValueError(
                "Pause transcription is not successful"
            )

        words, phrase_boundaries = load_words(
            transcript
        )

        chunks = create_chunks(words)

        llama = shutil.which(
            "llama-cli"
        )

        if llama is None:
            raise RuntimeError(
                "llama-cli not found in PATH"
            )

        chunk_results = []
        chunk_reports = []

        successful_chunks = 0
        fallback_chunks = 0

        for index, chunk in enumerate(chunks):
            chunk_number = index + 1

            set_status(
                job_dir,
                job_id,
                "processing",
                (
                    f"Refining transcript chunk "
                    f"{chunk_number}/{len(chunks)}"
                ),
                completed_chunks=index,
                total_chunks=len(chunks)
            )

            try:
                suggested_marks = analyze_chunk(
                    llama,
                    chunk,
                    job_dir,
                    chunk_number
                )

                successful_chunks += 1

                chunk_results.append(
                    suggested_marks
                )

                chunk_reports.append({
                    "chunk": chunk_number,
                    "status": "success",
                    "suggested_marks": len(
                        suggested_marks
                    ),
                    "raw_file": (
                        f"refinement_chunk_"
                        f"{chunk_number}_raw.txt"
                    )
                })

            except Exception as error:
                fallback_chunks += 1

                chunk_results.append({})

                chunk_reports.append({
                    "chunk": chunk_number,
                    "status": "fallback",
                    "reason": str(error),
                    "raw_file": (
                        f"refinement_chunk_"
                        f"{chunk_number}_raw.txt"
                    )
                })

        base_marks = build_base_marks(
            words
        )

        marks = apply_ai_marks(
            base_marks,
            chunks,
            chunk_results
        )

        marks = ensure_boundaries(
            words,
            marks,
            phrase_boundaries
        )

        marks = ensure_max_duration(
            words,
            marks,
            phrase_boundaries
        )

        sentences = assemble_sentences(
            words,
            marks
        )

        validate_sentences(
            words,
            sentences
        )

        if fallback_chunks == len(chunks):
            result_message = (
                "Completed using Whisper fallback; "
                "all Qwen chunks failed"
            )
        elif fallback_chunks > 0:
            result_message = (
                "Transcript refinement completed "
                "with partial fallback"
            )
        else:
            result_message = (
                "Transcript refinement completed"
            )

        result = {
            "status": "success",
            "job_id": job_id,
            "model": MODEL_NAME,
            "method": "chunked_refinement_v3",
            "language": transcript.get("language"),
            "start_seconds": transcript.get(
                "start_seconds"
            ),
            "total_words": len(words),
            "total_sentences": len(sentences),
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks,
            "fallback_chunks": fallback_chunks,
            "text": " ".join(
                sentence["text"]
                for sentence in sentences
            ),
            "sentences": sentences,
            "punctuation_marks": [
                {
                    "after": position,
                    "mark": mark
                }
                for position, mark in sorted(
                    marks.items()
                )
            ],
            "pauses": transcript.get(
                "pauses",
                []
            ),
            "source_file": "pause_transcript.json",
            "message": result_message
        }

        save_json(
            job_dir / "refined_transcript.json",
            result
        )

        save_json(
            job_dir / "refinement_debug.json",
            {
                "total_chunks": len(chunks),
                "successful_chunks": successful_chunks,
                "fallback_chunks": fallback_chunks,
                "chunks": chunk_reports
            }
        )

        set_status(
            job_dir,
            job_id,
            "completed",
            result_message,
            result=result
        )

    except Exception as error:
        print(
            f"Transcript refinement error: {error}"
        )

        try:
            set_status(
                job_dir,
                job_id,
                "failed",
                str(error)
            )

        except OSError as save_error:
            print(
                f"Status save error: {save_error}"
            )

    finally:
        with refinement_lock:
            if active_job == job_id:
                active_job = None


def start_refinement(job_dir, job_id):
    global active_job

    with refinement_lock:
        if active_job is not None:
            raise RuntimeError(
                "Another transcript refinement is running"
            )

        active_job = job_id

        try:
            initial_status = set_status(
                job_dir,
                job_id,
                "processing",
                "Transcript refinement queued"
            )

            worker = Thread(
                target=process_refinement,
                args=(job_dir, job_id),
                daemon=True
            )

            worker.start()

        except Exception:
            active_job = None
            raise

    return initial_status


def get_refinement_status(job_dir, job_id):
    path = (
        job_dir / "refinement_status.json"
    )

    if not path.is_file():
        return None

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if data.get("status") == "processing":
        with refinement_lock:
            running = active_job == job_id

        if not running:
            data = set_status(
                job_dir,
                job_id,
                "failed",
                "Refinement interrupted by server restart"
            )

    return data
