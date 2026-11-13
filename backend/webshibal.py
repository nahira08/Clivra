
import os
import re
import subprocess
import sys
import threading

from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.fusion_web import app


ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    ROOT_DIR
    / "backend"
    / "outputs"
    / "long_vods"
)

MAX_SEGMENTS = 3

# Configurable test position.
# Default is the beginning of the Twitch VOD.
START_SECONDS = int(
    os.environ.get(
        "CLIVRA_TEST_START_SECONDS",
        "0"
    )
)

if START_SECONDS < 0:
    raise ValueError(
        "CLIVRA_TEST_START_SECONDS cannot be negative."
    )

jobs = {}
jobs_lock = threading.Lock()


class AnalyzeRequest(BaseModel):
    url: str = Field(
        min_length=1,
        max_length=500
    )


def get_video_id(url):
    try:
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
            raise ValueError()

        match = re.fullmatch(
            r"/videos/([0-9]+)/?",
            parsed.path
        )

        if not match:
            raise ValueError()

        return match.group(1)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Please enter a valid HTTPS "
                "Twitch VOD URL."
            )
        )


def public_job(job):
    return {
        "task_id": job["task_id"],
        "job_id": job["job_id"],
        "video_id": job["video_id"],
        "status": job["status"],
        "stage": job["stage"],
        "stage_number": job["stage_number"],
        "total_stages": 8,
        "message": job["message"],
        "error": job["error"],
        "result_url": job["result_url"],
        "recent_logs": job["recent_logs"][-12:]
    }


def update_job(task_id, **changes):
    with jobs_lock:
        jobs[task_id].update(changes)


def run_pipeline(task_id, url, job_id):
    command = [
        sys.executable,
        "-u",
        "-m",
        "backend.pipeline",
        url,
        "--start",
        str(START_SECONDS),
        "--max-segments",
        str(MAX_SEGMENTS)
    ]

    process = None
    recent_logs = []

    try:
        update_job(
            task_id,
            status="running",
            stage="Starting pipeline",
            message="Preparing Clivra analysis."
        )

        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        stages = {
            1: "Downloading VOD",
            2: "Transcribing audio",
            3: "Finding humor candidates",
            4: "Reviewing humor with Qwen",
            5: "Collecting multimodal evidence",
            6: "Reviewing visual frames",
            7: "Analyzing audio",
            8: "Combining evidence"
        }

        for raw_line in process.stdout:
            line = raw_line.strip()

            if not line:
                continue

            print(
                f"[{task_id[:8]}] {line}",
                flush=True
            )

            recent_logs.append(line)
            recent_logs = recent_logs[-30:]

            match = re.search(
                r"\[([1-8])/8\]",
                line
            )

            if match:
                number = int(match.group(1))

                update_job(
                    task_id,
                    stage=stages[number],
                    stage_number=number,
                    message=line,
                    recent_logs=recent_logs.copy()
                )
            else:
                update_job(
                    task_id,
                    message=line,
                    recent_logs=recent_logs.copy()
                )

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                "Pipeline exited with code "
                f"{return_code}. "
                + (
                    recent_logs[-1]
                    if recent_logs
                    else "Check backend terminal."
                )
            )

        result_file = (
            OUTPUT_DIR
            / job_id
            / "evidence_fusion.json"
        )

        if not result_file.is_file():
            raise RuntimeError(
                "Evidence Fusion output was not created."
            )

        update_job(
            task_id,
            status="completed",
            stage="Completed",
            stage_number=8,
            message="Analysis completed successfully.",
            result_url=(
                "moments.html?job_id=" + job_id
            ),
            recent_logs=recent_logs.copy()
        )

    except Exception as error:
        update_job(
            task_id,
            status="failed",
            stage="Failed",
            message="Analysis failed.",
            error=str(error),
            recent_logs=recent_logs.copy()
        )

    finally:
        if (
            process is not None
            and process.poll() is None
        ):
            process.kill()
            process.wait()


@app.post("/api/pipeline/start")
def start_pipeline(request: AnalyzeRequest):
    url = request.url.strip()
    video_id = get_video_id(url)

    job_id = (
        f"vod_{video_id}_{START_SECONDS}_60"
    )

    with jobs_lock:
        active_job = next(
            (
                job
                for job in jobs.values()
                if job["status"] in {
                    "queued",
                    "running"
                }
            ),
            None
        )

        if active_job:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Another Clivra analysis "
                    "is currently running."
                )
            )

        task_id = uuid4().hex

        job = {
            "task_id": task_id,
            "job_id": job_id,
            "video_id": video_id,
            "status": "queued",
            "stage": "Queued",
            "stage_number": 0,
            "message": "Waiting to start.",
            "error": None,
            "result_url": None,
            "recent_logs": []
        }

        jobs[task_id] = job

    worker = threading.Thread(
        target=run_pipeline,
        args=(task_id, url, job_id),
        daemon=True
    )

    worker.start()

    return public_job(job)


@app.get("/api/pipeline/status/{task_id}")
def pipeline_status(task_id: str):
    with jobs_lock:
        job = jobs.get(task_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Task not found. It may have "
                    "been lost after server restart."
                )
            )

        return public_job(job)


# Apply CORS outside the existing FastAPI app
# to handle browser preflight requests.
app = CORSMiddleware(
    app,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600
)
