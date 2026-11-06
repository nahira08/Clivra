
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from uuid import uuid4, UUID
from functools import lru_cache
from urllib.parse import urlsplit
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from backend.vision import (
    start_visual_job,
    get_visual_job_status
)

import re
import sys
import subprocess
import json
import wave

app = FastAPI(
    title="Clivra API",
    description="AI-Powered Twitch Funny Moment Detection",
    version="0.5.0"
)

BASE_DIR = Path(__file__).resolve().parent 

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024

class TwitchURL(BaseModel):
    url: str

class TwitchPrepareRequest(BaseModel):
    url: str
    start_seconds: int = Field(default=0, ge=0)

class TranscribeRequest(BaseModel):
    job_id: UUID
    start_seconds: int = Field(ge=0)
    language: str | None = None

class VisualAnalysisRequest(BaseModel):
    job_id: UUID
    start_seconds: int = Field(default=0, ge=0)

def validate_twitch_url(url: str):

    try:
        parsed = urlsplit(url.strip())

        allowed_hosts = {
            "twitch.tv",
            "www.twitch.tv",
            "m.twitch.tv"
        }

        if (
            parsed.scheme != "https"
            or parsed.hostname not in allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            raise ValueError("Invalid Twitch domain")

        match = re.fullmatch(
            r"/videos/([0-9]+)/?",
            parsed.path
        )

        if not match:
            raise ValueError("Invalid Twitch VOD path")

        video_id = match.group(1)

        twitch_url = (
            f"https://www.twitch.tv/videos/{video_id}"
        )

        return video_id, twitch_url

    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid Twitch VOD URL"
        )

@app.get("/")
def home():
    return {
        "message": "Welcome to Clivra API!",
        "version": "0.5.0"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "success",
        "message": "Clivra backend is running!"
    }

@app.post("/api/videos/upload")
def upload_video(file: UploadFile = File(...)):

    if (
        not file.filename
        or not file.filename.lower().endswith(".mp4")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are allowed"
        )

    video_id = str(uuid4())
    saved_filename = f"{video_id}.mp4"
    file_path = UPLOAD_DIR / saved_filename

    file_size = 0

    try:
        with file_path.open("wb") as buffer:

            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds 100 MiB limit"
                    )

                buffer.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Video file is empty"
            )

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except OSError:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save video"
        )

    finally:
        file.file.close()

    return {
        "status": "success",
        "video_id": video_id,
        "original_filename": file.filename,
        "saved_filename": saved_filename,
        "size_bytes": file_size,
        "message": "Video uploaded successfully"
    }

@app.post("/api/twitch/info")
def get_twitch_info(data: TwitchURL):

    video_id, twitch_url = validate_twitch_url(
        data.url
    )

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 15,
        "retries": 1
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                twitch_url,
                download=False
            )

        if not info:
            raise HTTPException(
                status_code=404,
                detail="Twitch VOD not found"
            )

        return {
            "status": "success",
            "platform": "twitch",
            "video_id": video_id,
            "title": info.get("title"),
            "streamer": info.get("uploader"),
            "duration_seconds": info.get("duration"),
            "url": twitch_url,
            "message": "Twitch VOD information retrieved"
        }

    except DownloadError:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve Twitch VOD information"
        )
    
@app.post("/api/twitch/prepare")
def prepare_twitch_video(data: TwitchPrepareRequest):

    

    video_id, twitch_url = validate_twitch_url(
        data.url
    )

    # Durasi cuplikan yang akan diambil
    clip_duration = 60

    start = data.start_seconds
    end = start + clip_duration

    # Membuat folder khusus untuk pekerjaan
    job_id = str(uuid4())

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(
        job_dir / "clip.%(ext)s"
    )

    # Perintah yt-dlp untuk mengambil segmen
    download_command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--download-sections",
        f"*{start}-{end}",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        output_template,
        twitch_url
    ]

    try:
        # Mengambil video Twitch
        download_result = subprocess.run(
            download_command,
            capture_output=True,
            text=True,
            timeout=180
        )

        if download_result.returncode != 0:
            raise HTTPException(
                status_code=502,
                detail="Failed to retrieve Twitch video segment"
            )

        # Mencari hasil file video
        video_files = list(
            job_dir.glob("clip.*")
        )

        video_files = [
            path for path in video_files
            if path.suffix.lower() in {
                ".mp4", ".mkv", ".webm"
            }
        ]

        if not video_files:
            raise HTTPException(
                status_code=502,
                detail="Video segment was not created"
            )

        video_path = video_files[0]

        # Menentukan file audio output
        audio_path = job_dir / "audio.wav"

        # Mengekstrak audio menggunakan FFmpeg
        audio_command = [
            "ffmpeg",
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

        audio_result = subprocess.run(
            audio_command,
            capture_output=True,
            text=True,
            timeout=120
        )

        if (
            audio_result.returncode != 0
            or not audio_path.exists()
        ):
            raise HTTPException(
                status_code=500,
                detail="Failed to extract audio"
            )

        return {
            "status": "success",
            "job_id": job_id,
            "video_id": video_id,
            "start_seconds": start,
            "requested_end_seconds": end,
            "video_filename": video_path.name,
            "audio_filename": audio_path.name,
            "message": "Twitch video prepared successfully"
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Video processing timed out"
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Required processing tool not found"
        )

@lru_cache(maxsize=1)
def get_whisper_model():

    from faster_whisper import WhisperModel

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

@app.post("/api/ai/transcribe")
def transcribe_audio(data: TranscribeRequest):

    job_id = str(data.job_id)
    job_dir = OUTPUT_DIR / job_id
    audio_path = job_dir / "audio.wav"

    if not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found for this job"
        )

    if data.language is not None:
        if not re.fullmatch(r"[a-z]{2}", data.language):
            raise HTTPException(
                status_code=400,
                detail="Language must be a two-letter code"
            )

    try:
        model = get_whisper_model()

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language=data.language,
            vad_filter=True 
        )

        with wave.open(str(audio_path), "rb") as audio_file:
            audio_duration = (
                audio_file.getnframes() / audio_file.getframerate()
            )

        transcript = []

        for segment in segments:

            if not segment.text.strip():
                continue

            if segment.start >= audio_duration:
                continue

            local_start = round(max(0, segment.start), 2)
            local_end = round(min(segment.end, audio_duration), 2)

            if local_end <= local_start:
                continue

            transcript.append({
                "start": local_start,
                "end": local_end,
                "vod_start": round(
                    local_start + data.start_seconds, 2
                ),
                "vod_end": round(
                    local_end + data.start_seconds, 2
                ),
                "text": segment.text.strip()
            })

        result = {
            "status": "success",
            "job_id": job_id,
            "model": "faster-whisper-base",
            "language": info.language,
            "language_probability": round(
                info.language_probability, 3
            ),
            "start_seconds": data.start_seconds,
            "segments": transcript,
            "message": "Audio transcription completed"
        }

        output_path = job_dir / "transcript.json"

        with output_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=True
            )

        return result

    except (RuntimeError, ValueError, OSError) as error:
        print(f"Transcription error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Audio transcription failed"
        )

# ==========================================
# 11. START AI VISUAL ANALYSIS
# ==========================================

@app.post("/api/ai/analyze-visual", status_code=202)
def analyze_visual(data: VisualAnalysisRequest):

    job_id = str(data.job_id)
    job_dir = OUTPUT_DIR / job_id

    if not job_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Job folder not found"
        )

    if not (job_dir / "clip.mp4").is_file():
        raise HTTPException(
            status_code=404,
            detail="Video not found for this job"
        )

    try:
        return start_visual_job(
            job_dir=job_dir,
            job_id=job_id,
            start_seconds=data.start_seconds
        )

    except RuntimeError as error:
        if "Another visual analysis" in str(error):
            raise HTTPException(
                status_code=409,
                detail=str(error)
            )
        raise


# ==========================================
# 12. CHECK AI VISUAL STATUS
# ==========================================

@app.get("/api/ai/visual-status/{job_id}")
def visual_status(job_id: UUID):

    job_id_string = str(job_id)
    job_dir = OUTPUT_DIR / job_id_string

    if not job_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Job folder not found"
        )

    result = get_visual_job_status(
        job_dir,
        job_id_string
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No visual analysis found for this job"
        )

    return result