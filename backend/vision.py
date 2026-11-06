from pathlib import Path
from threading import Thread, Lock
from datetime import datetime, timezone
import json
import re
import shutil
import subprocess


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"

MAX_DURATION = 40
CONTEXT_SIZE = 8192
VIDEO_FPS = 0.1

job_lock = Lock()
active_job = None


def save_json(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary.replace(path)


def status_path(job_dir: Path):
    return job_dir / "visual_status.json"


def update_status(
    job_dir: Path,
    job_id: str,
    status: str,
    message: str,
    **extra
):
    data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra
    }

    save_json(status_path(job_dir), data)

    return data


def run_tool(command, timeout):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout
        )

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Processing timed out"
        ) from error

    except OSError as error:
        raise RuntimeError(
            f"Cannot run processing tool: {error}"
        ) from error

    if result.returncode != 0:
        details = (
            result.stderr or result.stdout
        ).strip()

        raise RuntimeError(
            f"Processing failed: {details[-800:]}"
        )

    return result


def prepare_video(job_dir: Path):
    source = job_dir / "clip.mp4"
    target = job_dir / "visual_40s.mp4"

    if not source.is_file():
        raise FileNotFoundError(
            "clip.mp4 not found"
        )

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found in PATH"
        )

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-t",
        str(MAX_DURATION),
        "-vf",
        "scale=448:-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        str(target)
    ]

    run_tool(command, timeout=180)

    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(
            "Prepared video is missing or empty"
        )

    return target


def extract_description(output: str, prompt: str):
    cleaned = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        output
    )

    media_marker = "Loaded media from"

    if media_marker not in cleaned:
        raise RuntimeError(
            "Qwen output does not contain the expected media marker"
        )

    cleaned = cleaned.split(
        media_marker,
        1
    )[1]

    cleaned = cleaned.split(
        "\n",
        1
    )[-1].strip()

    cleaned = re.sub(
        r"^>\s*",
        "",
        cleaned
    ).strip()

    cleaned = re.sub(
        r"^(?:User|Assistant):\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    ).strip()

    if prompt in cleaned:
        cleaned = cleaned.replace(
            prompt,
            "",
            1
        ).strip()

    cleaned = re.split(
        r"\bExiting\.\.\.",
        cleaned,
        maxsplit=1
    )[0].strip()

    cleaned = re.sub(
        r"\[\s*Prompt:.*?Generation:.*?\]",
        "",
        cleaned,
        flags=re.DOTALL
    ).strip()

    cleaned = re.sub(
        r"(?:\n\s*>\s*)+$",
        "",
        cleaned
    ).strip()

    if (
        not cleaned
        or cleaned in {">", "User:", "Assistant:"}
        or cleaned == prompt
    ):
        raise RuntimeError(
            "Qwen returned no usable description"
        )

    return cleaned


def process_visual_job(
    job_dir: Path,
    job_id: str,
    start_seconds: int
):
    global active_job

    try:
        update_status(
            job_dir,
            job_id,
            "processing",
            "Preparing video for Qwen3-VL"
        )

        video_path = prepare_video(job_dir)

        llama = shutil.which("llama-cli")

        if llama is None:
            raise RuntimeError(
                "llama-cli not found in PATH"
            )

        update_status(
            job_dir,
            job_id,
            "processing",
            "Qwen3-VL is analyzing the video"
        )

        prompt = (
            "Describe the main visual events in this "
            "VTuber video chronologically. "
            "Identify noticeable changes or unexpected "
            "visual events. Give approximate timestamps "
            "relative to the start of this clip. "
            "If nothing interesting happens, say so. "
            "Do not invent events or spoken dialogue."
        )

        command = [
            llama,
            "-hf",
            MODEL_NAME,
            "--video",
            str(video_path),
            "--video-fps",
            str(VIDEO_FPS),
            "-ngl",
            "0",
            "--no-mmproj-offload",
            "-c",
            str(CONTEXT_SIZE),
            "-n",
            "180",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "-p",
            prompt
        ]

        result = run_tool(
            command,
            timeout=600
        )

        raw_output_path = (
            job_dir / "visual_raw_output.txt"
        )

        raw_output_path.write_text(
            result.stdout,
            encoding="utf-8"
        )

        description = extract_description(
            result.stdout,
            prompt
        )

        description_path = (
            job_dir / "visual_description.txt"
        )

        description_path.write_text(
            description,
            encoding="utf-8"
        )

        analysis = {
            "status": "success",
            "job_id": job_id,
            "model": MODEL_NAME,
            "start_seconds": start_seconds,
            "analyzed_duration_seconds": MAX_DURATION,
            "video_filename": video_path.name,
            "description": description,
            "message": "Visual analysis completed"
        }

        save_json(
            job_dir / "visual_analysis.json",
            analysis
        )

        update_status(
            job_dir,
            job_id,
            "completed",
            "Visual analysis completed",
            result=analysis
        )

    except Exception as error:
        print(f"Visual job error: {error}")

        try:
            update_status(
                job_dir,
                job_id,
                "failed",
                str(error)
            )
        except OSError as status_error:
            print(
                f"Failed to save visual job status: "
                f"{status_error}"
            )

    finally:
        with job_lock:
            if active_job == job_id:
                active_job = None


def start_visual_job(
    job_dir: Path,
    job_id: str,
    start_seconds: int
):
    global active_job

    with job_lock:
        if active_job is not None:
            raise RuntimeError(
                "Another visual analysis is running"
            )

        active_job = job_id

        try:
            initial_status = update_status(
                job_dir,
                job_id,
                "processing",
                "Visual analysis queued"
            )

            worker = Thread(
                target=process_visual_job,
                args=(
                    job_dir,
                    job_id,
                    start_seconds
                ),
                daemon=True
            )

            worker.start()

        except Exception:
            active_job = None
            raise

    return initial_status


def get_visual_job_status(
    job_dir: Path,
    job_id: str
):
    path = status_path(job_dir)

    if not path.is_file():
        return None

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if data.get("status") == "processing":
        with job_lock:
            still_running = active_job == job_id

        if not still_running:
            data = update_status(
                job_dir,
                job_id,
                "failed",
                "Analysis interrupted by server restart"
            )

    return data