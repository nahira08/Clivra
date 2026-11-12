
import json
import re

from pathlib import Path

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.web_api import app


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "long_vods"
)

JOB_PATTERN = re.compile(
    r"vod_[0-9]+_[0-9]+_[0-9]+"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"]
)


@app.get("/api/long-vod/fusion/{job_id}")
def get_fusion_results(job_id: str):
    if not JOB_PATTERN.fullmatch(job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid VOD job ID."
        )

    output_path = (
        OUTPUT_DIR
        / job_id
        / "evidence_fusion.json"
    )

    if not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Evidence Fusion result not found. "
                "Run backend.evidence_fusion first."
            )
        )

    try:
        with output_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

    except (
        OSError,
        json.JSONDecodeError
    ) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot read fusion results: {error}"
        ) from error

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail="Invalid fusion result format."
        )

    return data