
import argparse
import json
import math
import re
import statistics

from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "long_vods"

MIN_RMS = 0.015
SPIKE_RATIO = 2.5


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


def to_number(value):
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def analyze_audio(moment):
    audio = moment.get("audio", {})

    if not isinstance(audio, dict):
        audio = {}

    raw_levels = audio.get("levels", [])

    if not isinstance(raw_levels, list):
        raw_levels = []

    levels = []

    for item in raw_levels:
        if not isinstance(item, dict):
            continue

        rms = to_number(item.get("rms"))
        peak = to_number(item.get("peak"))

        start = to_number(
            item.get("vod_start")
        )

        end = to_number(
            item.get("vod_end")
        )

        if (
            rms is None
            or peak is None
            or start is None
            or end is None
            or end <= start
            or rms < 0
            or peak < 0
        ):
            continue

        levels.append({
            "vod_start": start,
            "vod_end": end,
            "rms": rms,
            "peak": peak
        })

    levels.sort(
        key=lambda item: item["vod_start"]
    )

    if not levels:
        return {
            "status": "no_audio_data",
            "timestamp": moment.get("timestamp"),
            "windows_analyzed": 0,
            "median_rms": None,
            "maximum_rms": None,
            "maximum_peak": None,
            "maximum_ratio": None,
            "audio_energy_spike": False,
            "spike_windows": [],
            "classification": "unavailable",
            "interpretation": (
                "No usable audio measurements."
            ),
            "laughter_detected": None
        }

    rms_values = [
        item["rms"]
        for item in levels
    ]

    peak_values = [
        item["peak"]
        for item in levels
    ]

    median_rms = statistics.median(
        rms_values
    )

    maximum_rms = max(
        rms_values
    )

    maximum_peak = max(
        peak_values
    )

    baseline = max(
        median_rms,
        0.0001
    )

    maximum_ratio = (
        maximum_rms / baseline
    )

    spike_windows = []

    for item in levels:
        ratio = (
            item["rms"] / baseline
        )

        if (
            item["rms"] >= MIN_RMS
            and ratio >= SPIKE_RATIO
        ):
            spike_windows.append({
                "vod_start": item["vod_start"],
                "vod_end": item["vod_end"],
                "rms": item["rms"],
                "relative_ratio": round(
                    ratio,
                    2
                )
            })

    spike_detected = bool(
        spike_windows
    )

    if spike_detected:
        classification = (
            "sudden_energy_increase"
        )

        interpretation = (
            "One or more audio windows have "
            "substantially higher energy than "
            "the local median. The source of "
            "the increase is unknown."
        )

    elif maximum_rms < MIN_RMS:
        classification = (
            "low_audio_energy"
        )

        interpretation = (
            "Audio energy is relatively low "
            "in the analyzed interval."
        )

    else:
        classification = (
            "no_strong_energy_spike"
        )

        interpretation = (
            "No strong relative audio energy "
            "spike was detected in this interval."
        )

    return {
        "status": "completed",
        "timestamp": moment.get("timestamp"),
        "windows_analyzed": len(levels),
        "window_start": audio.get(
            "window_start"
        ),
        "window_end": audio.get(
            "window_end"
        ),
        "median_rms": round(
            median_rms,
            5
        ),
        "maximum_rms": round(
            maximum_rms,
            5
        ),
        "maximum_peak": round(
            maximum_peak,
            5
        ),
        "maximum_ratio": round(
            maximum_ratio,
            2
        ),
        "audio_energy_spike": (
            spike_detected
        ),
        "spike_windows": spike_windows,
        "classification": classification,
        "interpretation": interpretation,
        "laughter_detected": None
    }


def analyze_job(job_id):
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

    moments = evidence.get(
        "moments",
        []
    )

    if not isinstance(moments, list):
        raise ValueError(
            "Invalid multimodal evidence."
        )

    results = []

    for index, moment in enumerate(
        moments,
        start=1
    ):
        print(
            f"Analyzing audio "
            f"{index}/{len(moments)}: "
            f"{moment.get('id')}",
            flush=True
        )

        analysis = analyze_audio(
            moment
        )

        results.append({
            "id": moment.get("id"),
            "timestamp": moment.get(
                "timestamp"
            ),
            "trigger_text": moment.get(
                "trigger_text"
            ),
            "analysis": analysis
        })

        print(
            "  Classification:",
            analysis["classification"],
            flush=True
        )

        print(
            "  Energy spike:",
            analysis["audio_energy_spike"],
            flush=True
        )

    analyzed = sum(
        item["analysis"]["status"]
        == "completed"
        for item in results
    )

    if analyzed == len(results):
        status = "completed"
    else:
        status = "partial_failure"

    output = {
        "status": status,
        "job_id": job_id,
        "total_moments": len(results),
        "analyzed_moments": analyzed,
        "method": "relative_audio_energy",
        "laughter_detection_performed": False,
        "emotion_detection_performed": False,
        "humor_classification_performed": False,
        "moments": results,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    output_path = (
        job_dir / "audio_analysis.json"
    )

    save_json(
        output_path,
        output
    )

    return output, output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clivra Audio Energy Analysis"
        )
    )

    parser.add_argument(
        "job_id"
    )

    args = parser.parse_args()

    result, output_path = analyze_job(
        args.job_id
    )

    print()
    print("CLIVRA AUDIO ANALYSIS")
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
        "Analyzed:",
        result["analyzed_moments"]
    )

    print(
        "Saved to:",
        output_path
    )

    for item in result["moments"]:
        analysis = item["analysis"]

        print()
        print(
            "Timestamp:",
            item["timestamp"]
        )

        print(
            "Classification:",
            analysis["classification"]
        )

        print(
            "Median RMS:",
            analysis["median_rms"]
        )

        print(
            "Maximum RMS:",
            analysis["maximum_rms"]
        )

        print(
            "Energy spike:",
            analysis["audio_energy_spike"]
        )


if __name__ == "__main__":
    main()
