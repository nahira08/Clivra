
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import subprocess


OUTPUTS_DIR = Path("backend/outputs")
MIN_CLIP_SECONDS = 1.0


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def get_video_duration(video_path):
    ffprobe = shutil.which("ffprobe")

    if not ffprobe:
        raise RuntimeError(
            "FFprobe not found. Please install FFmpeg."
        )

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Unable to read video duration: "
            + result.stderr.strip()
        )

    return float(result.stdout.strip())


def get_vod_offset(job_dir):
    for filename in (
        "pause_transcript.json",
        "transcript.json"
    ):
        path = job_dir / filename

        if path.exists():
            data = load_json(path)

            return float(
                data.get("start_seconds", 0)
            )

    raise FileNotFoundError(
        "Transcript metadata not found. "
        "Cannot determine VOD offset."
    )


def validate_range(start, end, duration):
    start = float(start)
    end = float(end)

    if start < 0:
        raise ValueError(
            "Clip start cannot be negative."
        )

    if end <= start:
        raise ValueError(
            "Clip end must be greater than start."
        )

    if end - start < MIN_CLIP_SECONDS:
        raise ValueError(
            "Clip must be at least 1 second long."
        )

    if start >= duration:
        raise ValueError(
            "Clip start exceeds video duration."
        )

    if end > duration + 0.1:
        raise ValueError(
            f"Clip end {end:.2f}s exceeds "
            f"video duration {duration:.2f}s."
        )

    return round(start, 3), round(
        min(end, duration),
        3
    )


def load_candidates(job_dir):
    path = (
        job_dir / "context_funny_moments.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "context_funny_moments.json not found."
        )

    data = load_json(path)

    candidates = data.get(
        "candidates", []
    )

    if not candidates:
        raise ValueError(
            "No highlight candidates found. "
            "Use manual timestamps instead."
        )

    return candidates


def cut_video(
    source,
    destination,
    start,
    end
):
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found in PATH."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    duration = end - start

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(destination)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        destination.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "FFmpeg clipping failed: "
            + result.stderr[-1000:]
        )

    if (
        not destination.exists()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "FFmpeg did not generate a valid file."
        )


def next_filename(clips_dir):
    index = 1

    while True:
        filename = (
            f"highlight_{index:03d}.mp4"
        )

        destination = (
            clips_dir / filename
        )

        if not destination.exists():
            return destination

        index += 1


def export_highlights(
    job_id,
    start=None,
    end=None,
    vod_start=None,
    vod_end=None,
    candidate=None,
    outputs_dir=OUTPUTS_DIR
):
    job_dir = (
        Path(outputs_dir) / job_id
    )

    if not job_dir.is_dir():
        raise FileNotFoundError(
            f"Job not found: {job_id}"
        )

    source = job_dir / "clip.mp4"

    if not source.exists():
        raise FileNotFoundError(
            "clip.mp4 not found in job directory."
        )

    video_duration = get_video_duration(
        source
    )

    offset = get_vod_offset(
        job_dir
    )

    clips_dir = job_dir / "clips"

    clips_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    selections = []

    if start is not None and end is not None:
        selections.append({
            "start": start,
            "end": end,
            "source": "manual_relative"
        })

    elif (
        vod_start is not None
        and vod_end is not None
    ):
        selections.append({
            "start": vod_start - offset,
            "end": vod_end - offset,
            "source": "manual_vod"
        })

    elif candidate is not None:
        candidates = load_candidates(
            job_dir
        )

        if candidate == "all":
            indices = list(
                range(len(candidates))
            )
        else:
            number = int(candidate)

            if (
                number < 1
                or number > len(candidates)
            ):
                raise ValueError(
                    f"Candidate must be between "
                    f"1 and {len(candidates)}."
                )

            indices = [number - 1]

        for index in indices:
            item = candidates[index]

            if (
                "start" not in item
                or "end" not in item
            ):
                raise ValueError(
                    f"Candidate {index + 1} "
                    "has no valid timestamps."
                )

            selections.append({
                "start": item["start"],
                "end": item["end"],
                "source": "ai_candidate",
                "candidate_number": index + 1,
                "score": item.get("score"),
                "reason": item.get("reason")
            })

    else:
        raise ValueError(
            "Choose --start and --end, "
            "--vod-start and --vod-end, "
            "or --candidate."
        )

    exported = []

    for selection in selections:
        clip_start, clip_end = validate_range(
            selection["start"],
            selection["end"],
            video_duration
        )

        destination = next_filename(
            clips_dir
        )

        print(
            f"Exporting {destination.name}: "
            f"{clip_start:.2f}s - "
            f"{clip_end:.2f}s",
            flush=True
        )

        cut_video(
            source,
            destination,
            clip_start,
            clip_end
        )

        exported.append({
            "filename": destination.name,
            "path": str(destination),
            "start": clip_start,
            "end": clip_end,
            "vod_start": round(
                offset + clip_start,
                2
            ),
            "vod_end": round(
                offset + clip_end,
                2
            ),
            "duration": round(
                clip_end - clip_start,
                2
            ),
            "source": selection["source"],
            "candidate_number": selection.get(
                "candidate_number"
            ),
            "score": selection.get(
                "score"
            ),
            "reason": selection.get(
                "reason"
            )
        })

    result = {
        "status": "success",
        "job_id": job_id,
        "method": "ffmpeg_clipping_v1",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_file": "clip.mp4",
        "vod_offset": offset,
        "source_duration": round(
            video_duration,
            2
        ),
        "total_exported": len(exported),
        "clips": exported
    }

    save_json(
        clips_dir / "clipping_result.json",
        result
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export Clivra highlight clips "
            "using FFmpeg."
        )
    )

    parser.add_argument(
        "job_id"
    )

    parser.add_argument(
        "--outputs",
        default=str(OUTPUTS_DIR)
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--start",
        type=float
    )

    mode.add_argument(
        "--vod-start",
        type=float
    )

    mode.add_argument(
        "--candidate"
    )

    parser.add_argument(
        "--end",
        type=float
    )

    parser.add_argument(
        "--vod-end",
        type=float
    )

    args = parser.parse_args()

    if args.start is not None:
        if args.end is None:
            parser.error(
                "--start requires --end"
            )

        if args.vod_end is not None:
            parser.error(
                "Do not mix relative and VOD timestamps"
            )

    elif args.vod_start is not None:
        if args.vod_end is None:
            parser.error(
                "--vod-start requires --vod-end"
            )

        if args.end is not None:
            parser.error(
                "Do not mix relative and VOD timestamps"
            )

    elif (
        args.end is not None
        or args.vod_end is not None
    ):
        parser.error(
            "--candidate cannot be combined "
            "with manual timestamps"
        )

    result = export_highlights(
        job_id=args.job_id,
        start=args.start,
        end=args.end,
        vod_start=args.vod_start,
        vod_end=args.vod_end,
        candidate=args.candidate,
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
