
import argparse
import json
import re
import subprocess
import sys

from pathlib import Path
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "long_vods"

STAGES = [
    ("download", "manifest.json"),
    ("transcription", "merged_transcript.json"),
    ("detection", "funny_moments.json"),
    ("humor_review", "humor_review.json"),
    ("multimodal", "multimodal_evidence.json"),
    ("visual_review", "visual_review.json"),
    ("audio_analysis", "audio_analysis.json"),
    ("evidence_fusion", "evidence_fusion.json")
]


def extract_video_id(url):
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
        raise ValueError(
            "Please provide a valid HTTPS Twitch VOD URL."
        )

    match = re.fullmatch(
        r"/videos/([0-9]+)/?",
        parsed.path
    )

    if not match:
        raise ValueError(
            "Twitch URL must point to a VOD."
        )

    return match.group(1)


def find_model():
    cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-VL-2B-Instruct-GGUF"
        / "snapshots"
    )

    models = sorted(
        cache.glob(
            "*/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
        )
    )

    if not models:
        raise FileNotFoundError(
            "Qwen model not found in Hugging Face cache. "
            "Use --model to specify its location."
        )

    return models[-1]


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def run_command(command, label):
    print()
    print("=" * 55)
    print(label)
    print("=" * 55, flush=True)

    result = subprocess.run(
        command,
        cwd=BASE_DIR.parent,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code "
            f"{result.returncode}."
        )


def module_command(module, *args):
    return [
        sys.executable,
        "-m",
        module,
        *[str(value) for value in args]
    ]


def validate_output(path, label):
    if not path.is_file():
        raise RuntimeError(
            f"{label} did not create {path.name}."
        )

    data = read_json(path)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{label} produced invalid JSON."
        )

    status = data.get("status")

    if status in {
        "partial_failure",
        "failed",
        "error"
    }:
        raise RuntimeError(
            f"{label} reported status: {status}."
        )

    return data


def pipeline(
    url,
    start_seconds,
    max_segments,
    language,
    model,
    reuse_existing
):
    video_id = extract_video_id(url)

    job_id = (
        f"vod_{video_id}_{start_seconds}_60"
    )

    job_dir = OUTPUT_DIR / job_id

    model_path = (
        Path(model).expanduser().resolve()
        if model
        else find_model()
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print()
    print("CLIVRA AUTOMATED PIPELINE")
    print("=" * 55)
    print("Video ID:", video_id)
    print("Job ID:", job_id)
    print("Start:", start_seconds)
    print("Segment limit:", max_segments)
    print("Model:", model_path.name)

    manifest_path = (
        job_dir / "manifest.json"
    )

    transcript_path = (
        job_dir / "merged_transcript.json"
    )

    if (
        reuse_existing
        and manifest_path.is_file()
    ):
        print()
        print("[1/8] Reusing downloaded VOD segments.")

    else:
        run_command(
            module_command(
                "backend.long_vod",
                url,
                "--start",
                start_seconds,
                "--max-segments",
                max_segments
            ),
            "[1/8] Downloading VOD segments"
        )

    if not manifest_path.is_file():
        raise RuntimeError(
            "Download manifest was not created."
        )

    if (
        reuse_existing
        and transcript_path.is_file()
    ):
        print()
        print("[2/8] Reusing existing transcript.")

    else:
        run_command(
            module_command(
                "backend.long_vod_transcription",
                job_id,
                "--max-segments",
                max_segments,
                "--language",
                language
            ),
            "[2/8] Transcribing VOD"
        )

    if not transcript_path.is_file():
        raise RuntimeError(
            "Merged transcript was not created."
        )

    tasks = [
        (
            "detection",
            "[3/8] Finding humor candidates",
            module_command(
                "backend.funny_moments",
                job_id
            )
        ),
        (
            "humor_review",
            "[4/8] Reviewing humor with Qwen",
            module_command(
                "backend.humor_review",
                job_id,
                "--model",
                model_path
            )
        ),
        (
            "multimodal",
            "[5/8] Collecting visual and audio evidence",
            module_command(
                "backend.multimodal_evidence",
                job_id
            )
        ),
        (
            "visual_review",
            "[6/8] Reviewing frames with Qwen3-VL",
            module_command(
                "backend.visual_review",
                job_id,
                "--model",
                model_path
            )
        ),
        (
            "audio_analysis",
            "[7/8] Measuring audio changes",
            module_command(
                "backend.audio_analysis",
                job_id
            )
        ),
        (
            "evidence_fusion",
            "[8/8] Combining evidence",
            module_command(
                "backend.evidence_fusion",
                job_id
            )
        )
    ]

    output_names = dict(STAGES)

    for stage, label, command in tasks:
        output_path = (
            job_dir / output_names[stage]
        )

        if (
            reuse_existing
            and output_path.is_file()
        ):
            try:
                validate_output(
                    output_path,
                    label
                )

                print()
                print(
                    f"{label}: Reusing existing output."
                )

                continue

            except (
                ValueError,
                RuntimeError,
                json.JSONDecodeError
            ):
                print(
                    f"{label}: Existing result is "
                    f"invalid; running again."
                )

        run_command(command, label)

        validate_output(
            output_path,
            label
        )

    final_path = (
        job_dir / "evidence_fusion.json"
    )

    result = read_json(final_path)

    print()
    print("=" * 55)
    print("CLIVRA PIPELINE COMPLETED")
    print("=" * 55)

    print(
        "Candidates:",
        result.get("total_candidates", 0)
    )

    print(
        "Recommended:",
        result.get("recommended_count", 0)
    )

    print(
        "Needs review:",
        result.get("needs_review_count", 0)
    )

    print(
        "Rejected:",
        result.get("rejected_count", 0)
    )

    print("Output:", final_path)

    print()
    print("Open results:")
    print(
        "http://localhost:5500/moments.html"
        f"?job_id={job_id}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clivra automated VOD analysis pipeline"
        )
    )

    parser.add_argument(
        "url",
        help="HTTPS Twitch VOD URL"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0
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
        "--model",
        default=None
    )

    parser.add_argument(
        "--rerun",
        action="store_true"
    )

    args = parser.parse_args()

    if args.start < 0:
        parser.error(
            "--start cannot be negative."
        )

    if args.max_segments < 1:
        parser.error(
            "--max-segments must be at least 1."
        )

    try:
        pipeline(
            url=args.url,
            start_seconds=args.start,
            max_segments=args.max_segments,
            language=args.language,
            model=args.model,
            reuse_existing=not args.rerun
        )

    except Exception as error:
        print()
        print(
            "CLIVRA PIPELINE ERROR:",
            str(error)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
