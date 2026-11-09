
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import wave

from faster_whisper import WhisperModel


OUTPUT_DIR = Path("backend/outputs/long_vods")

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temporary.replace(path)


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def format_timestamp(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining:02d}"
    )


def get_audio_duration(audio_path):
    with wave.open(
        str(audio_path),
        "rb"
    ) as audio:
        return (
            audio.getnframes()
            / audio.getframerate()
        )


def create_model():
    print(
        "Loading Whisper model...",
        flush=True
    )

    return WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=4,
        num_workers=1
    )


def transcribe_segment(
    model,
    audio_path,
    job_id,
    segment,
    language
):
    segment_start = float(
        segment["start_seconds"]
    )

    audio_duration = get_audio_duration(
        audio_path
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False
    )

    transcript_segments = []
    transcript_words = []

    for item in segments:
        text = item.text.strip()

        if not text:
            continue

        local_start = max(
            0.0,
            float(item.start)
        )

        local_end = min(
            audio_duration,
            float(item.end)
        )

        if local_end <= local_start:
            continue

        word_list = []

        for word in item.words or []:
            word_text = word.word.strip()

            if not word_text:
                continue

            word_start = max(
                local_start,
                float(word.start)
            )

            word_end = min(
                local_end,
                float(word.end)
            )

            if word_end <= word_start:
                continue

            word_data = {
                "text": word_text,
                "start": round(
                    word_start,
                    2
                ),
                "end": round(
                    word_end,
                    2
                ),
                "vod_start": round(
                    segment_start + word_start,
                    2
                ),
                "vod_end": round(
                    segment_start + word_end,
                    2
                ),
                "probability": round(
                    float(word.probability),
                    4
                )
            }

            word_list.append(word_data)
            transcript_words.append(word_data)

        transcript_segments.append({
            "text": text,
            "start": round(
                local_start,
                2
            ),
            "end": round(
                local_end,
                2
            ),
            "vod_start": round(
                segment_start + local_start,
                2
            ),
            "vod_end": round(
                segment_start + local_end,
                2
            ),
            "vod_timestamp": format_timestamp(
                segment_start + local_start
            ),
            "words": word_list
        })

    return {
        "status": "success",
        "job_id": job_id,
        "segment_index": segment["index"],
        "start_seconds": segment_start,
        "end_seconds": segment["end_seconds"],
        "audio_duration": round(
            audio_duration,
            2
        ),
        "language": info.language,
        "requested_language": language,
        "model": f"faster-whisper-{MODEL_SIZE}",
        "total_segments": len(
            transcript_segments
        ),
        "total_words": len(
            transcript_words
        ),
        "segments": transcript_segments,
        "words": transcript_words,
        "text": " ".join(
            item["text"]
            for item in transcript_segments
        ),
        "generated_at": now()
    }


def valid_existing_transcript(
    path,
    segment,
    language
):
    if not path.is_file():
        return False

    try:
        data = load_json(path)

        return (
            data.get("status") == "success"
            and data.get("segment_index")
            == segment["index"]
            and data.get("start_seconds")
            == float(segment["start_seconds"])
            and data.get("requested_language")
            == language
            and isinstance(
                data.get("segments"),
                list
            )
            and isinstance(
                data.get("words"),
                list
            )
        )

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError
    ):
        return False


def merge_transcripts(
    job_dir,
    job_id,
    manifest,
    language
):
    all_segments = []
    all_words = []
    processed = []
    missing = []

    for segment in manifest["segments"]:
        index = segment["index"]

        transcript_path = (
            job_dir
            / "segments"
            / f"{index:04d}"
            / "transcript.json"
        )

        if not valid_existing_transcript(
            transcript_path,
            segment,
            language
        ):
            missing.append(index)
            continue

        data = load_json(
            transcript_path
        )

        all_segments.extend(
            data["segments"]
        )

        all_words.extend(
            data["words"]
        )

        processed.append(index)

    all_segments.sort(
        key=lambda item: item["vod_start"]
    )

    all_words.sort(
        key=lambda item: item["vod_start"]
    )

    merged = {
        "status": (
            "completed"
            if len(processed)
            == len(manifest["segments"])
            else "partial"
        ),
        "job_id": job_id,
        "video_id": manifest["video_id"],
        "title": manifest.get("title"),
        "streamer": manifest.get("streamer"),
        "vod_duration": manifest["vod_duration"],
        "model": f"faster-whisper-{MODEL_SIZE}",
        "language": language,
        "total_vod_segments": len(
            manifest["segments"]
        ),
        "transcribed_vod_segments": len(
            processed
        ),
        "processed_segment_indices": processed,
        "missing_segment_indices": missing,
        "total_speech_segments": len(
            all_segments
        ),
        "total_words": len(
            all_words
        ),
        "segments": all_segments,
        "words": all_words,
        "text": " ".join(
            item["text"]
            for item in all_segments
        ),
        "generated_at": now()
    }

    save_json(
        job_dir / "merged_transcript.json",
        merged
    )

    return merged


def process_transcription(
    job_id,
    max_segments=3,
    language="en",
    outputs_dir=OUTPUT_DIR
):
    if max_segments < 0:
        raise ValueError(
            "max_segments cannot be negative"
        )

    job_dir = (
        Path(outputs_dir)
        / job_id
    )

    manifest_path = (
        job_dir / "manifest.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Long VOD manifest.json not found"
        )

    manifest = load_json(
        manifest_path
    )

    available = [
        segment
        for segment in manifest["segments"]
        if segment["status"] == "completed"
    ]

    if max_segments == 0:
        selected = available
    else:
        selected = available[:max_segments]

    if not selected:
        raise ValueError(
            "No downloaded segments available"
        )

    print(
        f"Downloaded segments available: "
        f"{len(available)}"
    )

    print(
        f"Segments selected: "
        f"{len(selected)}"
    )

    pending = []

    for segment in selected:
        index = segment["index"]

        transcript_path = (
            job_dir
            / "segments"
            / f"{index:04d}"
            / "transcript.json"
        )

        if not valid_existing_transcript(
            transcript_path,
            segment,
            language
        ):
            pending.append(segment)

    model = None

    if pending:
        model = create_model()

    progress = {
        "job_id": job_id,
        "status": "processing",
        "language": language,
        "selected_segments": len(selected),
        "completed_segments": 0,
        "failed_segments": 0,
        "errors": [],
        "updated_at": now()
    }

    progress_path = (
        job_dir
        / "transcription_progress.json"
    )

    save_json(
        progress_path,
        progress
    )

    for segment in selected:
        index = segment["index"]

        segment_dir = (
            job_dir
            / "segments"
            / f"{index:04d}"
        )

        audio_path = (
            segment_dir / "audio.wav"
        )

        transcript_path = (
            segment_dir / "transcript.json"
        )

        if valid_existing_transcript(
            transcript_path,
            segment,
            language
        ):
            print(
                f"Segment {index}: "
                f"transcript already exists"
            )

            progress["completed_segments"] += 1

        else:
            print(
                f"Transcribing segment "
                f"{index}...",
                flush=True
            )

            try:
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio missing: "
                        f"{audio_path}"
                    )

                result = transcribe_segment(
                    model,
                    audio_path,
                    job_id,
                    segment,
                    language
                )

                save_json(
                    transcript_path,
                    result
                )

                progress["completed_segments"] += 1

                print(
                    f"Segment {index}: "
                    f"{result['total_words']} words, "
                    f"completed"
                )

            except Exception as error:
                progress["failed_segments"] += 1

                progress["errors"].append({
                    "segment_index": index,
                    "error": str(error)
                })

                print(
                    f"Segment {index}: "
                    f"failed - {error}"
                )

        progress["updated_at"] = now()

        save_json(
            progress_path,
            progress
        )

    merged = merge_transcripts(
        job_dir,
        job_id,
        manifest,
        language
    )

    progress["status"] = (
        "completed"
        if progress["failed_segments"] == 0
        else "partial_failure"
    )

    progress["updated_at"] = now()

    save_json(
        progress_path,
        progress
    )

    return {
        "status": progress["status"],
        "job_id": job_id,
        "selected_segments": len(selected),
        "completed_segments": (
            progress["completed_segments"]
        ),
        "failed_segments": (
            progress["failed_segments"]
        ),
        "merged_speech_segments": (
            merged["total_speech_segments"]
        ),
        "merged_words": merged["total_words"],
        "transcribed_vod_segments": (
            merged["transcribed_vod_segments"]
        ),
        "merged_file": str(
            job_dir / "merged_transcript.json"
        ),
        "progress_file": str(
            progress_path
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clivra Long VOD Whisper Transcription"
        )
    )

    parser.add_argument(
        "job_id"
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=3
    )

    parser.add_argument(
        "--language",
        default="en"
    )

    parser.add_argument(
        "--outputs",
        default=str(OUTPUT_DIR)
    )

    args = parser.parse_args()

    result = process_transcription(
        job_id=args.job_id,
        max_segments=args.max_segments,
        language=args.language,
        outputs_dir=args.outputs
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
