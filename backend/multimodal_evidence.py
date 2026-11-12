
import argparse
import json
import math
import shutil
import subprocess
import sys
import wave

from array import array
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "long_vods"

FRAME_OFFSETS = [-3, 0, 3]
AUDIO_WINDOW_BEFORE = 5
AUDIO_WINDOW_AFTER = 5
AUDIO_STEP_SECONDS = 0.5


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))

    return json.loads(
        path.read_text(encoding="utf-8")
    )


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


def format_time(seconds):
    total = max(0, int(seconds))

    hours = total // 3600
    minutes = (total % 3600) // 60
    remaining = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining:02d}"
    )


def parse_time(value):
    parts = str(value).split(":")

    if len(parts) != 3:
        raise ValueError("Invalid timestamp")

    hours, minutes, seconds = map(int, parts)

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def find_segment(manifest, vod_seconds):
    for segment in manifest["segments"]:
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])

        if start <= vod_seconds < end:
            return segment

    return None


def get_segment_file(job_dir, segment, filename):
    index = int(segment["index"])

    return (
        job_dir
        / "segments"
        / f"{index:04d}"
        / filename
    )


def extract_frame(
    video_path,
    local_seconds,
    destination
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{local_seconds:.3f}",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(destination)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-600:]
            or "FFmpeg failed"
        )

    if (
        not destination.is_file()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "Frame was not generated"
        )


def get_audio_levels(
    audio_path,
    local_start,
    local_end,
    vod_segment_start
):
    levels = []

    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()

        if sample_width != 2:
            raise ValueError(
                "Expected 16-bit PCM audio"
            )

        if channels != 1:
            raise ValueError(
                "Expected mono audio"
            )

        duration = (
            wav.getnframes() / sample_rate
        )

        start = max(
            0.0,
            min(local_start, duration)
        )

        end = max(
            start,
            min(local_end, duration)
        )

        current = start

        while current < end:
            next_time = min(
                current + AUDIO_STEP_SECONDS,
                end
            )

            first_frame = int(
                current * sample_rate
            )

            last_frame = int(
                next_time * sample_rate
            )

            wav.setpos(first_frame)

            raw = wav.readframes(
                max(0, last_frame - first_frame)
            )

            samples = array("h")
            samples.frombytes(raw)

            if sys.byteorder != "little":
                samples.byteswap()

            if samples:
                squares = sum(
                    int(sample) ** 2
                    for sample in samples
                )

                rms = math.sqrt(
                    squares / len(samples)
                ) / 32768.0

                peak = max(
                    abs(int(sample))
                    for sample in samples
                ) / 32768.0

            else:
                rms = 0.0
                peak = 0.0

            levels.append({
                "vod_start": round(
                    vod_segment_start + current,
                    2
                ),
                "vod_end": round(
                    vod_segment_start + next_time,
                    2
                ),
                "rms": round(rms, 5),
                "peak": round(peak, 5)
            })

            current = next_time

    return levels


def collect_audio(
    manifest,
    job_dir,
    trigger_seconds
):
    start = max(
        0.0,
        trigger_seconds - AUDIO_WINDOW_BEFORE
    )

    end = (
        trigger_seconds + AUDIO_WINDOW_AFTER
    )

    all_levels = []
    errors = []

    for segment in manifest["segments"]:
        segment_start = float(
            segment["start_seconds"]
        )

        segment_end = float(
            segment["end_seconds"]
        )

        if segment_end <= start:
            continue

        if segment_start >= end:
            continue

        audio_path = get_segment_file(
            job_dir,
            segment,
            "audio.wav"
        )

        if not audio_path.is_file():
            errors.append(
                f"Missing audio: segment {segment['index']}"
            )
            continue

        local_start = max(
            0.0,
            start - segment_start
        )

        local_end = min(
            segment_end - segment_start,
            end - segment_start
        )

        try:
            levels = get_audio_levels(
                audio_path,
                local_start,
                local_end,
                segment_start
            )

            all_levels.extend(levels)

        except Exception as error:
            errors.append(
                f"Segment {segment['index']}: {error}"
            )

    all_levels.sort(
        key=lambda item: item["vod_start"]
    )

    if all_levels:
        average_rms = sum(
            item["rms"]
            for item in all_levels
        ) / len(all_levels)

        maximum_peak = max(
            item["peak"]
            for item in all_levels
        )

    else:
        average_rms = None
        maximum_peak = None

    return {
        "window_start": round(start, 2),
        "window_end": round(end, 2),
        "average_rms": (
            round(average_rms, 5)
            if average_rms is not None
            else None
        ),
        "maximum_peak": (
            round(maximum_peak, 5)
            if maximum_peak is not None
            else None
        ),
        "levels": all_levels,
        "errors": errors,
        "interpretation": (
            "Audio energy only. "
            "Not a laughter or emotion classification."
        )
    }


def collect_frames(
    manifest,
    job_dir,
    trigger_seconds,
    output_dir
):
    frames = []

    for index, offset in enumerate(
        FRAME_OFFSETS,
        start=1
    ):
        vod_seconds = max(
            0.0,
            trigger_seconds + offset
        )

        segment = find_segment(
            manifest,
            vod_seconds
        )

        frame_info = {
            "index": index,
            "offset_seconds": offset,
            "vod_seconds": round(
                vod_seconds,
                2
            ),
            "vod_timestamp": format_time(
                vod_seconds
            ),
            "status": "missing",
            "file": None,
            "error": None
        }

        if segment is None:
            frame_info["error"] = (
                "No matching VOD segment"
            )

            frames.append(frame_info)
            continue

        video_path = get_segment_file(
            job_dir,
            segment,
            "clip.mp4"
        )

        if not video_path.is_file():
            frame_info["error"] = (
                "Video file is missing"
            )

            frames.append(frame_info)
            continue

        local_seconds = (
            vod_seconds
            - float(segment["start_seconds"])
        )

        destination = (
            output_dir
            / f"frame_{index:02d}.jpg"
        )

        try:
            extract_frame(
                video_path,
                local_seconds,
                destination
            )

            frame_info["status"] = "completed"
            frame_info["file"] = str(
                destination.relative_to(job_dir)
            )

        except Exception as error:
            frame_info["status"] = "error"
            frame_info["error"] = str(error)

        frames.append(frame_info)

    return frames


def collect_evidence(job_id):
    import re

    if not re.fullmatch(
        r"vod_[0-9]+_[0-9]+_[0-9]+",
        job_id
    ):
        raise ValueError("Invalid job ID")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH"
        )

    job_dir = OUTPUT_DIR / job_id

    manifest = read_json(
        job_dir / "manifest.json"
    )

    baseline = read_json(
        job_dir / "funny_moments.json"
    )

    reviews = read_json(
        job_dir / "humor_review.json"
    )

    if reviews.get("status") != "completed":
        raise ValueError(
            "AI humor review has not completed"
        )

    original_moments = {
        item["id"]: item
        for item in baseline.get("moments", [])
    }

    accepted = reviews.get(
        "accepted_moments",
        []
    )

    collected = []

    for index, review in enumerate(
        accepted,
        start=1
    ):
        moment_id = review["id"]

        original = original_moments.get(
            moment_id
        )

        if original is None:
            continue

        trigger_seconds = parse_time(
            original.get(
                "trigger_timestamp",
                original["timestamp"]
            )
        )

        output_dir = (
            job_dir
            / "multimodal"
            / moment_id
        )

        print(
            f"Collecting evidence "
            f"{index}/{len(accepted)}: {moment_id}",
            flush=True
        )

        frames = collect_frames(
            manifest,
            job_dir,
            trigger_seconds,
            output_dir
        )

        audio = collect_audio(
            manifest,
            job_dir,
            trigger_seconds
        )

        collected.append({
            "id": moment_id,
            "timestamp": format_time(
                trigger_seconds
            ),
            "trigger_seconds": trigger_seconds,
            "trigger_text": original.get(
                "trigger_text"
            ),
            "review_decision": (
                review["review"]["decision"]
            ),
            "frames": frames,
            "audio": audio
        })

    result = {
        "status": "completed",
        "job_id": job_id,
        "total_accepted_moments": len(accepted),
        "total_collected": len(collected),
        "visual_analysis_performed": False,
        "laughter_detection_performed": False,
        "moments": collected,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    output_path = (
        job_dir / "multimodal_evidence.json"
    )

    save_json(
        output_path,
        result
    )

    return result, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clivra Audio Visual Evidence Collector"
    )

    parser.add_argument("job_id")

    args = parser.parse_args()

    result, path = collect_evidence(
        args.job_id
    )

    print()
    print("CLIVRA MULTIMODAL EVIDENCE")
    print("=" * 40)

    print(
        "Accepted candidates:",
        result["total_accepted_moments"]
    )

    print(
        "Evidence collected:",
        result["total_collected"]
    )

    print("Output:", path)

    for moment in result["moments"]:
        completed = sum(
            frame["status"] == "completed"
            for frame in moment["frames"]
        )

        print()
        print(
            "Timestamp:",
            moment["timestamp"]
        )

        print(
            "Frames generated:",
            f"{completed}/{len(moment['frames'])}"
        )

        print(
            "Audio windows:",
            len(moment["audio"]["levels"])
        )


if __name__ == "__main__":
    main()
