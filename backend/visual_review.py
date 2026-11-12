
import argparse
import json
import os
import re
import shutil
import subprocess

from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "long_vods"

DEFAULT_TIMEOUT = 300

PROMPT = """
You are a visual evidence analyst for Clivra.

Describe ONLY what is directly visible in this Twitch VOD frame.

Do not assume that a smile means laughter.
Do not invent reactions, emotions, dialogue, or events.
Do not decide whether the scene is funny.
If something cannot be determined from one frame, say so.

Return exactly one JSON object with these keys:
{
  "scene": "Brief description of the game or scene",
  "visible_events": "Actions or events directly visible",
  "avatar_expression": "Visible avatar expression or unknown",
  "visual_evidence": "Relevant observations, without assumptions",
  "limitations": "What cannot be determined from this frame"
}

Return JSON only, without markdown.
""".strip()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
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


def resolve_binary(value):
    path = Path(value).expanduser()

    if path.is_file():
        return str(path.resolve())

    found = shutil.which(value)

    if found:
        return found

    raise FileNotFoundError(
        f"llama-cli not found: {value}"
    )


def find_model_files():
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

    projectors = sorted(
        cache.glob(
            "*/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
        )
    )

    if not models:
        raise FileNotFoundError(
            "Qwen3-VL Q4_K_M model not found."
        )

    model = models[-1]

    matching_projector = (
        model.parent
        / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    )

    if matching_projector.is_file():
        projector = matching_projector
    elif projectors:
        projector = projectors[-1]
    else:
        raise FileNotFoundError(
            "Qwen3-VL mmproj file not found."
        )

    return model, projector


def resolve_file(value, fallback):
    path = (
        Path(value).expanduser()
        if value
        else fallback
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return path.resolve()


def extract_json(output):
    decoder = json.JSONDecoder()
    matches = []

    for match in re.finditer(r"\{", output):
        try:
            value, _ = decoder.raw_decode(
                output[match.start():]
            )

        except json.JSONDecodeError:
            continue

        if (
            isinstance(value, dict)
            and "scene" in value
            and "avatar_expression" in value
        ):
            matches.append(value)

    if not matches:
        raise ValueError(
            "Qwen did not return valid visual JSON."
        )

    return matches[-1]


def normalize_review(raw):
    fields = [
        "scene",
        "visible_events",
        "avatar_expression",
        "visual_evidence",
        "limitations"
    ]

    review = {}

    for field in fields:
        value = raw.get(field, "")

        if isinstance(value, str):
            review[field] = value.strip()[:1000]
        else:
            review[field] = str(value)[:1000]

    if not review["scene"]:
        raise ValueError(
            "Visual review has no scene description."
        )

    return review


def run_visual_model(
    binary,
    model,
    mmproj,
    image_path,
    timeout
):
    command = [
        binary,
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--image",
        str(image_path),
        "-p",
        PROMPT,
        "-n",
        "300",
        "-c",
        "4096",
        "-t",
        "4",
        "-ngl",
        "0",
        "--no-mmproj-offload",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--temp",
        "0.1"
    ]

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

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Qwen visual inference timed out "
            f"after {timeout} seconds."
        ) from None

    except OSError as error:
        raise RuntimeError(
            f"Cannot start llama-cli: {error}"
        ) from error

    if result.returncode != 0:
        details = (
            result.stderr
            or result.stdout
            or "Unknown llama-cli error"
        )

        raise RuntimeError(
            f"llama-cli exited with code "
            f"{result.returncode}: "
            f"{details[-700:]}"
        )

    try:
        parsed = extract_json(result.stdout)

    except ValueError as error:
        raise ValueError(
            f"{error} "
            f"Output tail: {result.stdout[-700:]}"
        ) from error

    return normalize_review(parsed)


def get_safe_image_path(job_dir, relative_path):
    image_path = (
        job_dir / relative_path
    ).resolve()

    if not image_path.is_relative_to(
        job_dir.resolve()
    ):
        raise ValueError(
            "Image path is outside the VOD directory."
        )

    if image_path.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png"
    }:
        raise ValueError(
            "Unsupported image format."
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return image_path


def analyze_moment(
    moment,
    job_dir,
    binary,
    model,
    mmproj,
    timeout
):
    frame_results = []

    frames = moment.get("frames", [])

    for index, frame in enumerate(
        frames,
        start=1
    ):
        image_file = frame.get("file")

        entry = {
            "index": frame.get("index", index),
            "timestamp": frame.get(
                "vod_timestamp"
            ),
            "image_file": image_file,
            "status": "pending",
            "analysis": None,
            "error": None
        }

        if (
            frame.get("status") != "completed"
            or not image_file
        ):
            entry["status"] = "missing"
            entry["error"] = (
                "Frame was not generated."
            )

            frame_results.append(entry)
            continue

        try:
            image_path = get_safe_image_path(
                job_dir,
                image_file
            )

            print(
                f"  Analyzing frame "
                f"{index}/{len(frames)} "
                f"at {entry['timestamp']}...",
                flush=True
            )

            analysis = run_visual_model(
                binary,
                model,
                mmproj,
                image_path,
                timeout
            )

            entry["status"] = "completed"
            entry["analysis"] = analysis

            print(
                "  Visual analysis completed.",
                flush=True
            )

        except Exception as error:
            entry["status"] = "error"
            entry["error"] = str(error)[-1000:]

            print(
                f"  Error: {entry['error']}",
                flush=True
            )

        frame_results.append(entry)

    completed = [
        item
        for item in frame_results
        if item["status"] == "completed"
    ]

    descriptions = []

    for item in completed:
        analysis = item["analysis"]

        descriptions.append({
            "timestamp": item["timestamp"],
            "scene": analysis["scene"],
            "visible_events": (
                analysis["visible_events"]
            ),
            "avatar_expression": (
                analysis["avatar_expression"]
            ),
            "visual_evidence": (
                analysis["visual_evidence"]
            ),
            "limitations": (
                analysis["limitations"]
            )
        })

    return {
        "id": moment.get("id"),
        "timestamp": moment.get("timestamp"),
        "trigger_text": moment.get(
            "trigger_text"
        ),
        "frames_total": len(frame_results),
        "frames_analyzed": len(completed),
        "frame_reviews": frame_results,
        "visual_observations": descriptions,
        "humor_confirmed": False,
        "humor_confirmation_performed": False,
        "note": (
            "Frame descriptions are model-generated "
            "observations, not verified facts. "
            "Static frames cannot establish laughter, "
            "tone, or whether a joke succeeded."
        )
    }


def review_visuals(
    job_id,
    binary,
    model,
    mmproj,
    timeout=DEFAULT_TIMEOUT
):
    if not re.fullmatch(
        r"vod_[0-9]+_[0-9]+_[0-9]+",
        job_id
    ):
        raise ValueError(
            "Invalid long VOD job ID."
        )

    job_dir = OUTPUT_DIR / job_id

    evidence = read_json(
        job_dir / "multimodal_evidence.json"
    )

    moments = evidence.get("moments", [])

    if not isinstance(moments, list):
        raise ValueError(
            "Invalid multimodal evidence."
        )

    results = []

    for index, moment in enumerate(
        moments,
        start=1
    ):
        print()
        print(
            f"Visual review "
            f"{index}/{len(moments)}: "
            f"{moment.get('id')}",
            flush=True
        )

        result = analyze_moment(
            moment,
            job_dir,
            binary,
            model,
            mmproj,
            timeout
        )

        results.append(result)

    total_frames = sum(
        item["frames_total"]
        for item in results
    )

    analyzed_frames = sum(
        item["frames_analyzed"]
        for item in results
    )

    status = (
        "completed"
        if analyzed_frames == total_frames
        else "partial_failure"
    )

    output = {
        "status": status,
        "job_id": job_id,
        "model": model.name,
        "projector": mmproj.name,
        "total_moments": len(results),
        "total_frames": total_frames,
        "analyzed_frames": analyzed_frames,
        "visual_analysis_performed": True,
        "audio_interpretation_performed": False,
        "humor_confirmation_performed": False,
        "moments": results,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    output_path = (
        job_dir / "visual_review.json"
    )

    save_json(
        output_path,
        output
    )

    return output, output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clivra Qwen3-VL Visual Review"
        )
    )

    parser.add_argument(
        "job_id"
    )

    parser.add_argument(
        "--llama-cli",
        default=os.getenv(
            "CLIVRA_LLAMA_CLI",
            "llama-cli"
        )
    )

    parser.add_argument(
        "--model",
        default=os.getenv(
            "CLIVRA_MODEL_GGUF"
        )
    )

    parser.add_argument(
        "--mmproj",
        default=os.getenv(
            "CLIVRA_MMPROJ_GGUF"
        )
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT
    )

    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error(
            "--timeout must be positive."
        )

    default_model = None
    default_projector = None

    if not args.model or not args.mmproj:
        default_model, default_projector = (
            find_model_files()
        )

    model = resolve_file(
        args.model,
        default_model
    )

    mmproj = resolve_file(
        args.mmproj,
        default_projector
    )

    binary = resolve_binary(
        args.llama_cli
    )

    result, output_path = review_visuals(
        job_id=args.job_id,
        binary=binary,
        model=model,
        mmproj=mmproj,
        timeout=args.timeout
    )

    print()
    print("CLIVRA VISUAL REVIEW")
    print("=" * 40)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Moments:",
        result["total_moments"]
    )

    print(
        "Frames analyzed:",
        f"{result['analyzed_frames']}/"
        f"{result['total_frames']}"
    )

    print(
        "Saved to:",
        output_path
    )


if __name__ == "__main__":
    main()
