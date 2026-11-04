
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
import re

app = FastAPI(
    title="Clivra API",
    description="AI-Powered Twitch Funny Moment Detection",
    version="0.3.0"
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024


# Model data untuk input link Twitch
class TwitchURL(BaseModel):
    url: str

@app.get("/")
def home():
    return {
        "message": "Welcome to Clivra API!",
        "version": "0.3.0"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "success",
        "message": "Clivra backend is running!"
    }

@app.post("/api/videos/upload")
def upload_video(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".mp4"):
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

            while chunk := file.file.read(1024 * 1024):

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

    try:
        parsed_url = urlsplit(data.url)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid Twitch URL"
        )

    allowed_hosts = {
        "twitch.tv",
        "www.twitch.tv",
        "m.twitch.tv"
    }

    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in allowed_hosts
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.port is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid HTTPS Twitch URL"
        )

    match = re.fullmatch(
        r"/videos/([0-9]+)/?",
        parsed_url.path
    )

    if not match:
        raise HTTPException(
            status_code=400,
            detail="Please provide a Twitch VOD URL"
        )

    video_id = match.group(1)

    twitch_url = f"https://www.twitch.tv/videos/{video_id}"

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
                detail="Video information not found"
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
