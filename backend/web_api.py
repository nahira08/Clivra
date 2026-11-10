import json
import re
import subprocess
import sys
import threading

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.main import app
from backend.long_vod import validate_url


PROJECT_DIR = Path(__file__).resolve().parent.parent

LONG_VOD_DIR = (
    PROJECT_DIR
    / "backend"
    / "outputs"
    / "long_vods"
)

LONG_VOD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEGMENT_SECONDS = 60
MAX_SEGMENTS = 3

JOB_PATTERN = re.compile(
    r"^vod_[0-9]+_[0-9]+_60$"
)

active_jobs = set()
job_lock = threading.Lock()


# Allow the locally hosted frontend to use FastAPI

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class StartVODRequest(BaseModel):
    url: str
    start_seconds: int = Field(
        default=2410,
        ge=0
    )


def current_time():
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def get_job_dir(job_id):
    if not JOB_PATTERN.fullmatch(job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID"
        )

    return LONG_VOD_DIR / job_id


def get_state_path(job_dir):
    return job_dir / "web_job_status.json"


def save_state(
    job_dir,
    status,
    stage,
    message,
    error=None
):
    data = {
        "status": status,
        "stage": stage,
        "message": message,
        "error": error,
        "updated_at": current_time(),
    }

    save_json(
        get_state_path(job_dir),
        data
    )


def read_progress(job_dir, job_id):
    state_path = get_state_path(job_dir)

    if state_path.exists():
        state = read_json(state_path)
    else:
        state = {
            "status": "unknown",
            "stage": "unknown",
            "message": "Job has no status file",
            "error": None,
        }

    manifest_path = (
        job_dir / "manifest.json"
    )

    transcription_path = (
        job_dir
        / "transcription_progress.json"
    )

    merged_path = (
        job_dir
        / "merged_transcript.json"
    )

    download_completed = 0
    download_failed = 0
    transcription_completed = 0
    transcription_failed = 0

    if manifest_path.exists():
        manifest = read_json(manifest_path)

        selected = manifest.get(
            "segments",
            []
        )[:MAX_SEGMENTS]

        download_completed = sum(
            segment.get("status") == "completed"
            for segment in selected
        )

        download_failed = sum(
            segment.get("status") == "failed"
            for segment in selected
        )

    if transcription_path.exists():
        progress = read_json(
            transcription_path
        )

        transcription_completed = int(
            progress.get(
                "completed_segments",
                0
            )
        )

        transcription_failed = int(
            progress.get(
                "failed_segments",
                0
            )
        )

    # The API does not consider a partial file
    # an automatically completed web job.

    return {
        "job_id": job_id,
        **state,
        "download_completed": download_completed,
        "download_failed": download_failed,
        "transcription_completed": transcription_completed,
        "transcription_failed": transcription_failed,
        "total_segments": MAX_SEGMENTS,
        "result_available": (
            merged_path.is_file()
            and state.get("status") == "completed"
        ),
    }


def run_command(command, timeout):
    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

    if result.returncode != 0:
        details = (
            result.stderr
            or result.stdout
            or "Unknown processing error"
        )

        raise RuntimeError(
            details[-2000:]
        )

    return result.stdout


def process_web_job(
    job_id,
    url,
    start_seconds
):
    job_dir = get_job_dir(job_id)

    try:
        save_state(
            job_dir,
            "processing",
            "downloading",
            "Downloading Twitch VOD segments..."
        )

        download_command = [
            sys.executable,
            "-m",
            "backend.long_vod",
            url,
            "--start",
            str(start_seconds),
            "--segment-seconds",
            str(SEGMENT_SECONDS),
            "--max-segments",
            str(MAX_SEGMENTS),
        ]

        run_command(
            download_command,
            timeout=1800
        )

        manifest_path = (
            job_dir / "manifest.json"
        )

        if not manifest_path.is_file():
            raise RuntimeError(
                "VOD manifest was not created"
            )

        manifest = read_json(
            manifest_path
        )

        selected = manifest.get(
            "segments",
            []
        )[:MAX_SEGMENTS]

        if len(selected) != MAX_SEGMENTS:
            raise RuntimeError(
                "VOD has fewer than 3 segments "
                "from the selected start time"
            )

        if not all(
            segment.get("status") == "completed"
            for segment in selected
        ):
            failed = [
                str(segment.get("error"))
                for segment in selected
                if segment.get("status") != "completed"
            ]

            raise RuntimeError(
                "Download failed: "
                + "; ".join(failed)
            )

        save_state(
            job_dir,
            "processing",
            "transcribing",
            "Transcribing downloaded audio..."
        )

        transcription_command = [
            sys.executable,
            "-m",
            "backend.long_vod_transcription",
            job_id,
            "--max-segments",
            str(MAX_SEGMENTS),
            "--language",
            "en",
            "--outputs",
            str(LONG_VOD_DIR),
        ]

        run_command(
            transcription_command,
            timeout=3600
        )

        transcription_progress_path = (
            job_dir
            / "transcription_progress.json"
        )

        if not transcription_progress_path.is_file():
            raise RuntimeError(
                "Transcription progress not found"
            )

        transcription_progress = read_json(
            transcription_progress_path
        )

        if (
            transcription_progress.get(
                "failed_segments",
                0
            ) != 0
            or transcription_progress.get(
                "completed_segments",
                0
            ) != MAX_SEGMENTS
        ):
            raise RuntimeError(
                "Some selected segments "
                "were not transcribed"
            )

        merged_path = (
            job_dir
            / "merged_transcript.json"
        )

        if not merged_path.is_file():
            raise RuntimeError(
                "Merged transcript not found"
            )

        save_state(
            job_dir,
            "completed",
            "finished",
            "VOD transcription completed"
        )

    except Exception as error:
        save_state(
            job_dir,
            "failed",
            "error",
            "VOD processing failed",
            error=str(error)[:2000]
        )

    finally:
        with job_lock:
            active_jobs.discard(job_id)


@app.post("/api/long-vod/start")
def start_long_vod(
    data: StartVODRequest,
    background_tasks: BackgroundTasks
):
    try:
        video_id, clean_url = validate_url(
            data.url
        )

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    job_id = (
        f"vod_{video_id}_"
        f"{data.start_seconds}_"
        f"{SEGMENT_SECONDS}"
    )

    job_dir = get_job_dir(job_id)

    with job_lock:
        if job_id in active_jobs:
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Job is already running"
            }

        active_jobs.add(job_id)

    try:
        job_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_state(
            job_dir,
            "queued",
            "queued",
            "Waiting for processing to start"
        )

        background_tasks.add_task(
            process_web_job,
            job_id,
            clean_url,
            data.start_seconds
        )

    except Exception:
        with job_lock:
            active_jobs.discard(job_id)

        raise

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job accepted",
        "total_segments": MAX_SEGMENTS
    }


@app.get("/api/long-vod/status/{job_id}")
def long_vod_status(job_id: str):
    job_dir = get_job_dir(job_id)

    if not job_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    state_path = get_state_path(job_dir)

    if not state_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Web job status not found"
        )

    result = read_progress(
        job_dir,
        job_id
    )

    # After an application restart, background
    # tasks are no longer running.

    if result["status"] in {
        "queued",
        "processing"
    }:
        with job_lock:
            running = job_id in active_jobs

        if not running:
            result["status"] = "interrupted"
            result["message"] = (
                "Server was restarted during "
                "processing. Submit the URL "
                "again to resume."
            )

    return result


@app.get("/api/long-vod/result/{job_id}")
def long_vod_result(job_id: str):
    job_dir = get_job_dir(job_id)

    state_path = get_state_path(job_dir)

    if not state_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    state = read_json(state_path)

    if state.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Job is not completed"
        )

    merged_path = (
        job_dir
        / "merged_transcript.json"
    )

    if not merged_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Transcript result not found"
        )

    return read_json(merged_path)


@app.get("/api/long-vod/health")
def long_vod_health():
    return {
        "status": "ok",
        "service": "Clivra Long VOD Web API",
        "segment_seconds": SEGMENT_SECONDS,
        "max_segments": MAX_SEGMENTS,
    }
