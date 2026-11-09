
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit
import argparse
import json
import math
import re
import shutil
import subprocess

from yt_dlp import YoutubeDL


OUTPUT_DIR = Path("backend/outputs/long_vods")
DEFAULT_SEGMENT_SECONDS = 60
DEFAULT_MAX_SEGMENTS = 3
DOWNLOAD_TIMEOUT = 300
AUDIO_TIMEOUT = 120


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary.replace(path)


def validate_url(url):
    parsed = urlsplit(url.strip())

    if (
        parsed.scheme != "https"
        or parsed.hostname not in {
            "twitch.tv",
            "www.twitch.tv",
            "m.twitch.tv"
        }
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise ValueError("Invalid Twitch URL")

    match = re.fullmatch(
        r"/videos/([0-9]+)/?",
        parsed.path
    )

    if not match:
        raise ValueError("Invalid Twitch VOD path")

    video_id = match.group(1)

    return (
        video_id,
        f"https://www.twitch.tv/videos/{video_id}"
    )


def get_video_info(url):
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 20,
        "retries": 2,
        "noplaylist": True
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(
            url,
            download=False
        )

    if not info:
        raise RuntimeError(
            "Unable to retrieve Twitch VOD metadata"
        )

    duration = info.get("duration")

    if duration is None:
        raise ValueError(
            "Twitch VOD duration is unavailable"
        )

    duration = float(duration)

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            "Invalid Twitch VOD duration"
        )

    return {
        "title": info.get("title"),
        "streamer": info.get("uploader"),
        "duration": duration
    }


def run_command(command, timeout):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )

    if result.returncode != 0:
        detail = (
            result.stderr or result.stdout
        ).strip()

        raise RuntimeError(
            detail[-1200:]
            or "Command failed"
        )


def download_segment(
    url,
    segment_dir,
    start,
    end
):
    segment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for item in segment_dir.glob("clip.*"):
        if item.is_file():
            item.unlink()

    output_template = str(
        segment_dir / "clip.%(ext)s"
    )

    command = [
        "yt-dlp",
        "--no-playlist",
        "--download-sections",
        f"*{start:.3f}-{end:.3f}",
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        output_template,
        url
    ]

    run_command(
        command,
        DOWNLOAD_TIMEOUT
    )

    candidates = [
        path
        for path in segment_dir.glob("clip.*")
        if path.suffix.lower() in {
            ".mp4",
            ".mkv",
            ".webm"
        }
    ]

    if not candidates:
        raise RuntimeError(
            "Video segment was not created"
        )

    source = candidates[0]
    destination = segment_dir / "clip.mp4"

    if source != destination:
        run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                str(destination)
            ],
            DOWNLOAD_TIMEOUT
        )

        source.unlink(missing_ok=True)

    if (
        not destination.is_file()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "Downloaded video is empty"
        )

    return destination


def extract_audio(video_path, segment_dir):
    audio_path = segment_dir / "audio.wav"

    audio_path.unlink(missing_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path)
    ]

    run_command(
        command,
        AUDIO_TIMEOUT
    )

    if (
        not audio_path.is_file()
        or audio_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Audio extraction failed"
        )

    return audio_path


def create_manifest(
    video_id,
    url,
    video_info,
    start_seconds,
    segment_seconds
):
    duration = video_info["duration"]
    segments = []

    index = 1
    current = float(start_seconds)

    while current < duration:
        end = min(
            current + segment_seconds,
            duration
        )

        segments.append({
            "index": index,
            "start_seconds": round(current, 3),
            "end_seconds": round(end, 3),
            "status": "pending",
            "video_file": None,
            "audio_file": None,
            "error": None
        })

        current = end
        index += 1

    return {
        "status": "pending",
        "video_id": video_id,
        "url": url,
        "title": video_info["title"],
        "streamer": video_info["streamer"],
        "vod_duration": duration,
        "start_seconds": start_seconds,
        "segment_seconds": segment_seconds,
        "total_segments": len(segments),
        "completed_segments": 0,
        "failed_segments": 0,
        "created_at": now(),
        "updated_at": now(),
        "segments": segments
    }


def update_manifest(manifest, path):
    segments = manifest["segments"]

    completed = sum(
        item["status"] == "completed"
        for item in segments
    )

    failed = sum(
        item["status"] == "failed"
        for item in segments
    )

    manifest["completed_segments"] = completed
    manifest["failed_segments"] = failed
    manifest["updated_at"] = now()

    if completed == len(segments):
        manifest["status"] = "completed"
    elif failed:
        manifest["status"] = "partial_failure"
    elif completed:
        manifest["status"] = "processing"
    else:
        manifest["status"] = "pending"

    save_json(path, manifest)


def process_long_vod(
    url,
    start_seconds=0,
    segment_seconds=DEFAULT_SEGMENT_SECONDS,
    max_segments=DEFAULT_MAX_SEGMENTS
):
    if start_seconds < 0:
        raise ValueError(
            "Start seconds cannot be negative"
        )

    if segment_seconds < 10:
        raise ValueError(
            "Segment duration must be at least 10 seconds"
        )

    if max_segments < 0:
        raise ValueError(
            "Max segments cannot be negative"
        )

    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp was not found in PATH"
        )

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg was not found in PATH"
        )

    video_id, clean_url = validate_url(url)

    job_name = (
        f"vod_{video_id}_"
        f"{start_seconds}_{segment_seconds}"
    )

    job_dir = OUTPUT_DIR / job_name
    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = job_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if manifest.get("url") != clean_url:
            raise ValueError(
                "Existing job has a different URL"
            )

        print("Resuming existing VOD job...")

    else:
        print("Retrieving Twitch VOD information...")

        info = get_video_info(clean_url)

        if start_seconds >= info["duration"]:
            raise ValueError(
                "Start time exceeds VOD duration"
            )

        manifest = create_manifest(
            video_id,
            clean_url,
            info,
            start_seconds,
            segment_seconds
        )

        update_manifest(
            manifest,
            manifest_path
        )

    segments = manifest["segments"]

    if max_segments == 0:
        selected = segments
    else:
        selected = segments[:max_segments]

    for segment in selected:
        index = segment["index"]

        segment_dir = (
            job_dir
            / "segments"
            / f"{index:04d}"
        )

        video_path = segment_dir / "clip.mp4"
        audio_path = segment_dir / "audio.wav"

        if (
            segment["status"] == "completed"
            and video_path.is_file()
            and audio_path.is_file()
            and video_path.stat().st_size > 0
            and audio_path.stat().st_size > 0
        ):
            print(
                f"Segment {index}: already completed"
            )
            continue

        print(
            f"Processing segment {index}/"
            f"{len(segments)} "
            f"({segment['start_seconds']:.1f}s - "
            f"{segment['end_seconds']:.1f}s)",
            flush=True
        )

        segment["status"] = "processing"
        segment["error"] = None

        update_manifest(
            manifest,
            manifest_path
        )

        try:
            video = download_segment(
                clean_url,
                segment_dir,
                segment["start_seconds"],
                segment["end_seconds"]
            )

            audio = extract_audio(
                video,
                segment_dir
            )

            segment["video_file"] = str(
                video.relative_to(job_dir)
            )

            segment["audio_file"] = str(
                audio.relative_to(job_dir)
            )

            segment["status"] = "completed"

            print(
                f"Segment {index}: completed"
            )

        except Exception as error:
            segment["status"] = "failed"
            segment["error"] = str(error)[:1000]

            print(
                f"Segment {index}: failed - {error}"
            )

            update_manifest(
                manifest,
                manifest_path
            )

            break

        update_manifest(
            manifest,
            manifest_path
        )

    update_manifest(
        manifest,
        manifest_path
    )

    result = {
        "status": manifest["status"],
        "job_id": job_name,
        "vod_duration": manifest["vod_duration"],
        "total_segments": manifest["total_segments"],
        "completed_segments": manifest["completed_segments"],
        "failed_segments": manifest["failed_segments"],
        "manifest_file": str(manifest_path),
        "job_directory": str(job_dir)
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Clivra Long Twitch VOD Processor"
    )

    parser.add_argument(
        "url",
        help="Twitch VOD URL"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0
    )

    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=DEFAULT_SEGMENT_SECONDS
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=DEFAULT_MAX_SEGMENTS
    )

    args = parser.parse_args()

    result = process_long_vod(
        url=args.url,
        start_seconds=args.start,
        segment_seconds=args.segment_seconds,
        max_segments=args.max_segments
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
