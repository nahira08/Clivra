
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import re
import shutil
import subprocess


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"

WINDOW_SECONDS = 14.0
STEP_SECONDS = 7.0

PRE_ROLL = 5.0
POST_ROLL = 2.5

MIN_AI_SCORE = 70
MAX_CANDIDATES = 5

MODEL_TIMEOUT = 300
MAX_OUTPUT_TOKENS = 220
CONTEXT_SIZE = 4096


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def normalize_tokens(text):
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        str(text).lower()
    )


def load_transcript(job_dir):
    path = job_dir / "pause_transcript.json"

    if path.exists():
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        words = []

        for phrase in data.get("phrases", []):
            for item in phrase.get("words", []):
                text = str(
                    item.get("text", "")
                ).strip()

                if not text:
                    continue

                words.append({
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "text": text
                })

        if words:
            words.sort(
                key=lambda word: word["start"]
            )

            return data, words, path.name

    path = job_dir / "transcript.json"

    if not path.exists():
        raise FileNotFoundError(
            "No transcript file found"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    words = []

    for segment in data.get("segments", []):
        text = str(
            segment.get("text", "")
        ).strip()

        if not text:
            continue

        words.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": text
        })

    if not words:
        raise ValueError(
            "Transcript contains no timed speech"
        )

    return data, words, path.name


def load_visual_summary(job_dir):
    path = job_dir / "visual_analysis.json"

    if not path.exists():
        return ""

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    def search(node, depth=0):
        if depth > 5:
            return ""

        if isinstance(node, dict):
            for key in (
                "description",
                "visual_description",
                "summary",
                "text"
            ):
                value = node.get(key)

                if (
                    isinstance(value, str)
                    and len(value.strip()) > 10
                ):
                    return value.strip()

            for value in node.values():
                found = search(value, depth + 1)

                if found:
                    return found

        return ""

    return search(data)[:160]


def create_windows(words):
    duration = max(
        word["end"]
        for word in words
    )

    windows = []
    current = 0.0

    while current < duration:
        boundary = min(
            current + WINDOW_SECONDS,
            duration
        )

        selected = [
            word
            for word in words
            if (
                word["end"] > current
                and word["start"] < boundary
            )
        ]

        if selected:
            windows.append({
                "start": selected[0]["start"],
                "end": selected[-1]["end"],
                "words": selected,
                "text": " ".join(
                    word["text"]
                    for word in selected
                )
            })

        if boundary >= duration:
            break

        current += STEP_SECONDS

    return windows


def build_prompt(window, visual):
    return (
        "Judge whether this VTuber conversation "
        "contains a clear joke, witty reply, or "
        "genuinely amusing contradiction.\n"
        "Be skeptical: greetings, repetition, "
        "ordinary chat, and speech recognition "
        "errors are not automatically jokes.\n"
        "Do not invent events or reactions.\n"
        "Return a single compact JSON object "
        "with exactly these fields:\n"
        "funny: boolean\n"
        "score: integer from 0 to 100\n"
        "reason: explanation of at most 12 words\n"
        "anchor: an exact short phrase copied "
        "from the transcript containing "
        "the punchline\n"
        "For ordinary talk, use funny=false, "
        "score below 50, and empty anchor.\n"
        "No markdown. No extra explanation.\n"
        f"Visual background: {visual or 'none'}\n"
        f"Transcript: {window['text']}\n"
        "JSON:"
    )


def clean_model_output(output):
    text = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        output
    )

    if "... (truncated)" in text:
        text = text.rsplit(
            "... (truncated)",
            1
        )[-1]

    if "[ Prompt:" in text:
        text = text.split(
            "[ Prompt:",
            1
        )[0]

    if "Exiting..." in text:
        text = text.split(
            "Exiting...",
            1
        )[0]

    return text.strip()


def parse_decision(output):
    cleaned = clean_model_output(output)
    decoder = json.JSONDecoder()
    candidates = []

    for match in re.finditer(r"\{", cleaned):
        try:
            item, _ = decoder.raw_decode(
                cleaned[match.start():]
            )
        except json.JSONDecodeError:
            continue

        if not isinstance(item, dict):
            continue

        if not isinstance(
            item.get("funny"),
            bool
        ):
            continue

        score = item.get("score")

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            continue

        reason = item.get("reason")
        anchor = item.get("anchor")

        if not isinstance(reason, str):
            continue

        if not isinstance(anchor, str):
            continue

        candidates.append({
            "funny": item["funny"],
            "score": max(
                0,
                min(100, int(score))
            ),
            "reason": reason.strip()[:180],
            "anchor": anchor.strip()[:180]
        })

    if not candidates:
        raise ValueError(
            "Qwen did not return a complete "
            "valid decision JSON"
        )

    return candidates[-1]


def ask_qwen(
    llama,
    window,
    visual,
    job_dir=None,
    window_number=None
):
    prompt = build_prompt(
        window,
        visual
    )

    command = [
        llama,
        "-hf",
        MODEL_NAME,
        "-ngl",
        "0",
        "--no-mmproj-offload",
        "-c",
        str(CONTEXT_SIZE),
        "-n",
        str(MAX_OUTPUT_TOKENS),
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "-p",
        prompt
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=MODEL_TIMEOUT
    )

    if job_dir is None:
        debug_path = Path(
            "backend/outputs/context_qwen_raw.txt"
        )
    else:
        debug_path = (
            Path(job_dir)
            / f"context_window_{window_number}_raw.txt"
        )

    debug_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    debug_path.write_text(
        "========== STDOUT ==========\n"
        + result.stdout
        + "\n\n========== STDERR ==========\n"
        + result.stderr,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                result.stderr
                or result.stdout
            )[-600:]
        )

    decision = parse_decision(
        result.stdout
    )

    return decision, result.stdout


def find_anchor(words, phrase):
    target = normalize_tokens(phrase)

    if len(target) < 2:
        return None

    tokens = []
    token_to_word = []

    for index, word in enumerate(words):
        pieces = normalize_tokens(
            word["text"]
        )

        for piece in pieces:
            tokens.append(piece)
            token_to_word.append(index)

    size = len(target)

    for start in range(
        len(tokens) - size + 1
    ):
        if tokens[start:start + size] != target:
            continue

        first = token_to_word[start]
        last = token_to_word[
            start + size - 1
        ]

        return {
            "start": words[first]["start"],
            "end": words[last]["end"],
            "text": " ".join(
                word["text"]
                for word in words[first:last + 1]
            )
        }

    return None


def overlap_ratio(a, b):
    overlap = max(
        0.0,
        min(a["end"], b["end"])
        - max(a["start"], b["start"])
    )

    shorter = min(
        a["end"] - a["start"],
        b["end"] - b["start"]
    )

    if shorter <= 0:
        return 0.0

    return overlap / shorter


def is_duplicate(candidate, existing):
    if overlap_ratio(
        candidate,
        existing
    ) >= 0.65:
        return True

    if abs(
        candidate["anchor_start"]
        - existing["anchor_start"]
    ) <= 3.0:
        return True

    return False


def build_candidate(
    decision,
    anchor,
    window,
    vod_offset,
    duration
):
    start = max(
        0.0,
        anchor["start"] - PRE_ROLL
    )

    end = min(
        duration,
        anchor["end"] + POST_ROLL
    )

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "vod_start": round(
            vod_offset + start,
            2
        ),
        "vod_end": round(
            vod_offset + end,
            2
        ),
        "anchor_start": round(
            anchor["start"],
            2
        ),
        "anchor_end": round(
            anchor["end"],
            2
        ),
        "anchor": anchor["text"],
        "score": decision["score"],
        "reason": decision["reason"],
        "transcript": window["text"],
        "method": "qwen_context_v2",
        "needs_review": True
    }


def detect_highlights(
    job_dir,
    job_id,
    only_window=None
):
    job_dir = Path(job_dir)

    if not job_dir.exists():
        raise FileNotFoundError(
            f"Job directory not found: {job_dir}"
        )

    transcript, words, source_file = (
        load_transcript(job_dir)
    )

    visual = load_visual_summary(
        job_dir
    )

    windows = create_windows(
        words
    )

    if only_window is not None:
        if (
            only_window < 1
            or only_window > len(windows)
        ):
            raise ValueError(
                f"Window must be between 1 "
                f"and {len(windows)}"
            )

        selected_windows = [
            (only_window, windows[only_window - 1])
        ]
    else:
        selected_windows = list(
            enumerate(windows, 1)
        )

    llama = shutil.which(
        "llama-cli"
    )

    if not llama:
        raise RuntimeError(
            "llama-cli not found in PATH"
        )

    vod_offset = float(
        transcript.get(
            "start_seconds",
            0
        )
    )

    duration = max(
        word["end"]
        for word in words
    )

    evaluated = []
    candidates = []

    for index, window in selected_windows:
        print(
            f"Analyzing window "
            f"{index}/{len(windows)}...",
            flush=True
        )

        report = {
            "window": index,
            "start": round(
                window["start"],
                2
            ),
            "end": round(
                window["end"],
                2
            ),
            "transcript": window["text"],
            "status": "pending"
        }

        try:
            decision, raw_output = ask_qwen(
                llama,
                window,
                visual,
                job_dir=job_dir,
                window_number=index
            )

            report["status"] = "success"
            report["decision"] = decision
            report["model_response"] = (
                clean_model_output(
                    raw_output
                )[:2500]
            )

            if not decision["funny"]:
                report["selection"] = "not_funny"
                evaluated.append(report)
                continue

            if (
                decision["score"]
                < MIN_AI_SCORE
            ):
                report["selection"] = (
                    "below_threshold"
                )
                evaluated.append(report)
                continue

            anchor = find_anchor(
                window["words"],
                decision["anchor"]
            )

            if anchor is None:
                report["selection"] = (
                    "anchor_not_found"
                )
                evaluated.append(report)
                continue

            candidate = build_candidate(
                decision,
                anchor,
                window,
                vod_offset,
                duration
            )

            candidates.append(candidate)

            report["selection"] = "candidate"
            report["anchor_match"] = anchor

        except (
            ValueError,
            RuntimeError,
            subprocess.TimeoutExpired,
            OSError
        ) as error:
            report["status"] = "failed"
            report["error"] = str(error)[:600]

        evaluated.append(report)

    candidates.sort(
        key=lambda candidate: (
            -candidate["score"],
            candidate["start"]
        )
    )

    unique = []

    for candidate in candidates:
        if any(
            is_duplicate(
                candidate,
                existing
            )
            for existing in unique
        ):
            continue

        unique.append(candidate)

        if len(unique) >= MAX_CANDIDATES:
            break

    unique.sort(
        key=lambda candidate: (
            candidate["vod_start"]
        )
    )

    valid_windows = sum(
        report["status"] == "success"
        for report in evaluated
    )

    failed_windows = sum(
        report["status"] == "failed"
        for report in evaluated
    )

    result = {
        "status": "success",
        "job_id": job_id,
        "method": "context_highlights_v2",
        "model": MODEL_NAME,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_file": source_file,
        "visual_available": bool(visual),
        "total_windows": len(windows),
        "analyzed_windows": len(selected_windows),
        "valid_windows": valid_windows,
        "failed_windows": failed_windows,
        "total_candidates": len(unique),
        "candidates": unique,
        "message": (
            "AI-generated highlight candidates. "
            "Manual review is recommended."
        )
    }

    debug_data = {
        "method": "context_highlights_v2",
        "only_window": only_window,
        "windows": evaluated
    }

    if only_window is not None:
        result_path = (
            job_dir
            / f"context_funny_moments_test_{only_window}.json"
        )

        debug_path = (
            job_dir
            / f"context_highlights_debug_test_{only_window}.json"
        )

    else:
        result_path = (
            job_dir
            / "context_funny_moments.json"
        )

        debug_path = (
            job_dir
            / "context_highlights_debug.json"
        )

    save_json(
        result_path,
        result
    )

    save_json(
        debug_path,
        debug_data
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "job_id"
    )

    parser.add_argument(
        "--outputs",
        default="backend/outputs"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=None
    )

    args = parser.parse_args()

    job_dir = (
        Path(args.outputs)
        / args.job_id
    )

    result = detect_highlights(
        job_dir,
        args.job_id,
        only_window=args.window
    )

    print(
        json.dumps(
            {
                "status": result["status"],
                "method": result["method"],
                "total_windows": result["total_windows"],
                "analyzed_windows": result["analyzed_windows"],
                "valid_windows": result["valid_windows"],
                "failed_windows": result["failed_windows"],
                "total_candidates": result["total_candidates"],
                "candidates": result["candidates"]
            },
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
