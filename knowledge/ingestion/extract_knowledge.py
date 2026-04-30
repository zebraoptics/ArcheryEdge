"""
knowledge/extract_knowledge.py
================================
Offline tool — run on Mac or cloud, NOT on Jetson.

Extracts structured archery coaching knowledge from YouTube videos
using the Gemini API, producing validated JSON files ready for indexing.

Workflow
--------
1. Provide a list of YouTube URLs in a text file or via CLI
2. Gemini watches each video and extracts coaching entries
3. Each entry is validated against the schema
4. Output: one JSON file per video + a merged knowledge_base.json

Usage
-----
  # Single video
  python knowledge/extract_knowledge.py \
      --url "https://www.youtube.com/watch?v=XXXX" \
      --output data/knowledge_json

  # Batch from a text file (one URL per line)
  python knowledge/extract_knowledge.py \
      --url-file data/coaching_videos.txt \
      --output data/knowledge_json

  # Merge all JSON files into one knowledge base
  python knowledge/extract_knowledge.py \
      --merge-only \
      --output data/knowledge_json

Requirements
------------
  pip install google-genai python-dotenv
  GEMINI_API_KEY in .env or environment
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("extract_knowledge")

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are an expert archery biomechanics analyst and coaching knowledge extractor.

Your task is to watch the provided archery coaching video and extract structured
coaching knowledge entries in the exact JSON format specified below.

## Context

This knowledge base will be used by an AI coaching system that:
1. Measures 3D joint angles from stereo cameras observing a real archer
2. Detects the shooting phase (setup / draw / anchor / aim / release / follow_through)
3. Queries this knowledge base with measured angles to find matching technique errors
4. Generates coaching feedback using a small language model (Qwen2.5-3B) on a
   NVIDIA Jetson Orin Nano Super edge device

The archer disciplines covered are: recurve, barebow.
Do NOT include compound bow technique (different biomechanics).

## Shooting phases

- setup         : stance, nocking, pre-draw position, bow raise
- draw          : pulling string from setup to full draw
- anchor        : string hand at full draw position against face/jaw
- aim           : holding period between anchor and release
- release       : letting the string go
- follow_through: body position maintained after release

## Measurable joints and metrics

Use ONLY these joint names (exactly as written):
  draw_elbow, draw_shoulder, draw_wrist, draw_hand
  bow_elbow, bow_shoulder, bow_wrist
  spine, head, hip, knee

Use ONLY these metric names (exactly as written):
  draw_elbow_angle            degrees  — angle at draw elbow joint (3D)
  bow_arm_extension_angle     degrees  — bow arm elbow extension (3D)
  draw_shoulder_height_diff   mm       — draw shoulder height relative to bow shoulder
  spine_lateral_tilt          degrees  — spine tilt left/right from vertical
  spine_forward_lean          degrees  — spine lean forward/back from vertical
  head_tilt                   degrees  — head tilt from vertical
  hip_rotation                degrees  — pelvis rotation relative to target line
  anchor_point_height         mm       — draw hand height relative to jaw/chin
  anchor_point_consistency    mm       — std deviation of anchor across shots
  follow_through_duration     seconds  — time body holds position post-release
  draw_shoulder_rise          mm       — shoulder rise from setup to anchor

## Required JSON output format

Return a JSON object with this EXACT structure.
Extract 5–15 entries per video — one entry per distinct error or technique point.
ALL fields are required. Use "" for unknown strings, 0 for unknown numbers, [] for
unknown arrays. Do NOT add extra fields.

{
  "source_video": {
    "title": "<video title>",
    "url": "<youtube url>",
    "coach_name": "<coach name or empty string>",
    "discipline": "<recurve or barebow>",
    "skill_level_target": "<beginner / intermediate / advanced / all>"
  },
  "entries": [
    {
      "id": "<unique: discipline_phase_shortname_NNN  e.g. rec_draw_elbow_collapse_001>",
      "phase": "<setup / draw / anchor / aim / release / follow_through>",
      "category": "<technique_error / technique_cue / drill / principle>",
      "joints_involved": ["<joint name>"],
      "metric": {
        "name": "<metric name from list, or empty string if not directly measurable>",
        "observable_range": [<min>, <max>],
        "ideal_range": [<min>, <max>],
        "unit": "<degrees / mm / seconds>",
        "measurement_axis": "<3d_sagittal / 3d_frontal / 3d_transverse / spatial>"
      },
      "error_name": "<5 words max — name of the error or technique point>",
      "description": "<1-2 sentences: biomechanical explanation of why this matters>",
      "root_causes": ["<cause>", "<cause>"],
      "observable_signs": ["<what 3D pose data would show>"],
      "coaching_cues": ["<verbal cue 1>", "<verbal cue 2>", "<verbal cue 3>"],
      "drill": {
        "name": "<drill name>",
        "description": "<how to perform, 2-3 sentences>",
        "reps": <integer>,
        "focus_metric": "<metric name or empty string>"
      },
      "severity": "<low / medium / high>",
      "score_impact": "<1 sentence: how this error affects arrow placement on target>",
      "related_entries": [],
      "embed_text": "<flat keyword string: phase + joints + metric + range + error name + cues + score impact — optimised for semantic similarity search>"
    }
  ]
}

## Rules

1. Only extract knowledge observable from 3D pose data.
   Exclude anything needing EMG, force plates, or equipment sensors.

2. embed_text must be a single flat string — no JSON inside it.
   Write it as dense keywords: phase, joint names, metric name, value ranges,
   error name, key coaching cues, score impact consequences.

3. observable_range = what a problematic archer looks like.
   ideal_range = what the coach recommends.
   If the video gives no exact numbers, estimate from biomechanics literature
   and note "estimated" at the end of the description field.

4. Each entry covers exactly ONE error or cue. Do not combine.

5. coaching_cues must be short verbal phrases a coach says out loud —
   suitable for text-to-speech audio cues during live training.

6. For category "drill", metric.name may be empty string and ranges may be 0.

7. related_entries should list IDs of other entries in this same response
   that are biomechanically linked. Leave as [] if none.

Return ONLY the JSON object. No preamble, no explanation, no markdown code fences.
"""

# ─────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────

VALID_PHASES = {
    "setup", "draw", "anchor", "aim", "release", "follow_through"
}
VALID_CATEGORIES = {
    "technique_error", "technique_cue", "drill", "principle"
}
VALID_SEVERITIES = {"low", "medium", "high"}
VALID_DISCIPLINES = {"recurve", "barebow"}
VALID_METRICS = {
    "draw_elbow_angle", "bow_arm_extension_angle",
    "draw_shoulder_height_diff", "spine_lateral_tilt",
    "spine_forward_lean", "head_tilt", "hip_rotation",
    "anchor_point_height", "anchor_point_consistency",
    "follow_through_duration", "draw_shoulder_rise", "",
}
VALID_JOINTS = {
    "draw_elbow", "draw_shoulder", "draw_wrist", "draw_hand",
    "bow_elbow", "bow_shoulder", "bow_wrist",
    "spine", "head", "hip", "knee",
}


def _warn(msg: str) -> None:
    log.warning("  SCHEMA: %s", msg)


def validate_entry(entry: dict, idx: int) -> list[str]:
    """
    Validate one entry dict. Returns list of warning strings.
    Does not raise — caller decides whether to reject.
    """
    warnings = []
    prefix = f"entry[{idx}] id={entry.get('id', '?')}"

    # Required string fields
    for field in ("id", "phase", "category", "error_name",
                  "description", "severity", "score_impact", "embed_text"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            warnings.append(f"{prefix}: missing or empty '{field}'")

    if entry.get("phase") not in VALID_PHASES:
        warnings.append(f"{prefix}: invalid phase '{entry.get('phase')}'")

    if entry.get("category") not in VALID_CATEGORIES:
        warnings.append(f"{prefix}: invalid category '{entry.get('category')}'")

    if entry.get("severity") not in VALID_SEVERITIES:
        warnings.append(f"{prefix}: invalid severity '{entry.get('severity')}'")

    # joints_involved
    joints = entry.get("joints_involved", [])
    if not isinstance(joints, list) or len(joints) == 0:
        warnings.append(f"{prefix}: joints_involved is empty")
    for j in joints:
        if j not in VALID_JOINTS:
            warnings.append(f"{prefix}: unknown joint '{j}'")

    # metric
    metric = entry.get("metric", {})
    if not isinstance(metric, dict):
        warnings.append(f"{prefix}: metric is not a dict")
    else:
        if metric.get("name", "") not in VALID_METRICS:
            warnings.append(
                f"{prefix}: unknown metric name '{metric.get('name')}'"
            )
        for rng in ("observable_range", "ideal_range"):
            r = metric.get(rng, [])
            if not isinstance(r, list) or len(r) != 2:
                warnings.append(f"{prefix}: metric.{rng} must be [min, max]")
            # [0, 0] is acceptable as a placeholder when range is not quantifiable

    # lists
    for field in ("root_causes", "observable_signs",
                  "coaching_cues", "related_entries"):
        if not isinstance(entry.get(field), list):
            warnings.append(f"{prefix}: '{field}' must be a list")

    # coaching_cues — need at least 1
    cues = entry.get("coaching_cues", [])
    if isinstance(cues, list) and len(cues) == 0:
        warnings.append(f"{prefix}: coaching_cues is empty")

    # drill — name only required when category IS "drill"
    drill = entry.get("drill", {})
    if not isinstance(drill, dict):
        warnings.append(f"{prefix}: drill is not a dict")
    else:
        if entry.get("category") == "drill" and not drill.get("name"):
            warnings.append(f"{prefix}: drill.name is empty (required for category=drill)")
        if not isinstance(drill.get("reps"), int):
            warnings.append(f"{prefix}: drill.reps must be int")

    # embed_text length — warn if suspiciously short
    et = entry.get("embed_text", "")
    if isinstance(et, str) and len(et) < 30:
        warnings.append(f"{prefix}: embed_text is very short ({len(et)} chars)")

    return warnings


def validate_response(data: dict) -> tuple[bool, list[str]]:
    """
    Full validation of a Gemini response dict.
    Returns (is_valid, list_of_warnings).
    """
    all_warnings = []

    if "source_video" not in data:
        all_warnings.append("Missing top-level 'source_video'")
    else:
        sv = data["source_video"]
        if sv.get("discipline") not in VALID_DISCIPLINES:
            all_warnings.append(
                f"source_video.discipline '{sv.get('discipline')}' not in "
                f"{VALID_DISCIPLINES} — defaulting to 'recurve'"
            )
            data["source_video"]["discipline"] = "recurve"

    if "entries" not in data or not isinstance(data["entries"], list):
        all_warnings.append("Missing or invalid 'entries' array — fatal")
        return False, all_warnings

    if len(data["entries"]) == 0:
        all_warnings.append("entries array is empty — fatal")
        return False, all_warnings

    for i, entry in enumerate(data["entries"]):
        entry_warnings = validate_entry(entry, i)
        all_warnings.extend(entry_warnings)

    # Check for duplicate IDs
    ids = [e.get("id", "") for e in data["entries"]]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        all_warnings.append(f"Duplicate entry IDs: {dupes}")

    # Fatal only if > 60% of entries have warnings — soft issues are acceptable
    entries_with_warnings = sum(
        1 for i, e in enumerate(data["entries"])
        if validate_entry(e, i)
    )
    pct = entries_with_warnings / len(data["entries"])
    if pct > 0:
        log.debug(
            "  %d/%d entries have warnings (%.0f%%)",
            entries_with_warnings, len(data["entries"]), pct * 100,
        )
    fatal = pct > 0.60

    return not fatal, all_warnings


# ─────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────

def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _repair_truncated_json(raw: str) -> str:
    """
    Attempt to repair a truncated JSON string by closing any open
    structures. Handles the common case where the response was cut off
    mid-string or mid-object due to token limits.
    """
    # Remove the incomplete trailing token (unterminated string / object)
    # Walk back from the end to the last complete entry boundary
    # Strategy: find the last complete entry by locating the last "}," or "}"
    # before the truncation point, then close the array and root object.

    # First try: just parse as-is in case it's a minor whitespace issue
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Find the last safely closed entry — last occurrence of "}," or "}\n"
    # that is followed only by incomplete content
    last_good = -1
    for match in re.finditer(r'\}(?:\s*,|\s*\n)', raw):
        last_good = match.end()

    if last_good == -1:
        return raw  # can't repair

    truncated = raw[:last_good].rstrip().rstrip(',')

    # Count unclosed brackets/braces to figure out what to close
    opens  = truncated.count('{') - truncated.count('}')
    arrays = truncated.count('[') - truncated.count(']')

    closing = ''
    # Close the entries array if it was open
    if arrays > 0:
        closing += ']' * arrays
    # Close any open objects
    if opens > 0:
        closing += '}' * opens

    repaired = truncated + closing
    try:
        json.loads(repaired)
        log.warning("Repaired truncated JSON — removed incomplete trailing entry")
        return repaired
    except json.JSONDecodeError:
        return raw  # repair failed, return original for error reporting


def extract_from_url(
    client: genai.Client,
    url: str,
    model_name: str = "gemini-2.5-flash",
    retries: int = 2,
) -> Optional[dict]:
    """
    Call Gemini to extract coaching knowledge from one YouTube URL.
    Returns parsed dict or None on failure.
    """
    log.info("Extracting from: %s", url)

    contents = [
        EXTRACTION_PROMPT,
        f"Please extract archery coaching knowledge from this video: {url}",
    ]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=65536,
    )

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            raw = response.text.strip()

            # Strip accidental markdown fences Gemini sometimes adds
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            # Attempt to repair truncated JSON before parsing
            raw = _repair_truncated_json(raw)

            data = json.loads(raw)

            # Inject URL if Gemini left it empty
            if "source_video" in data and not data["source_video"].get("url"):
                data["source_video"]["url"] = url

            is_valid, warnings = validate_response(data)
            for w in warnings:
                log.warning("  %s", w)

            if is_valid:
                log.info(
                    "  Extracted %d entries (%d warnings)",
                    len(data["entries"]), len(warnings),
                )
                return data
            else:
                log.error(
                    "  Validation failed on attempt %d/%d",
                    attempt + 1, retries + 1,
                )
                if attempt < retries:
                    log.info("  Retrying in 5s…")
                    time.sleep(5)

        except json.JSONDecodeError as e:
            log.error("  JSON parse error (attempt %d): %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(5)
        except Exception as e:
            log.error("  Gemini API error (attempt %d): %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(10)

    log.error("  All attempts failed for %s", url)
    return None


# ─────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    """Convert YouTube URL to a safe filename stem."""
    # Extract video ID if present
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if match:
        return f"video_{match.group(1)}"
    # Fallback: sanitise the full URL
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", url)[-60:]
    return f"video_{safe}"


def save_entry_file(data: dict, out_dir: Path) -> Path:
    """Save one video's extraction to a JSON file. Returns path."""
    stem = url_to_filename(data["source_video"].get("url", "unknown"))
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("  Saved → %s", path)
    return path


def merge_knowledge_base(json_dir: Path) -> dict:
    """
    Merge all per-video JSON files into a single knowledge_base.json.
    Deduplicates by entry ID (last-seen wins).
    """
    files = sorted(json_dir.glob("video_*.json"))
    if not files:
        log.error("No video_*.json files found in %s", json_dir)
        return {}

    merged_entries = {}
    sources = []

    for fp in files:
        try:
            data = json.loads(fp.read_text())
            sources.append(data.get("source_video", {}))
            for entry in data.get("entries", []):
                eid = entry.get("id", "")
                if eid in merged_entries:
                    log.warning(
                        "Duplicate ID '%s' from %s — overwriting", eid, fp.name
                    )
                merged_entries[eid] = entry
        except Exception as e:
            log.error("Failed to read %s: %s", fp.name, e)

    kb = {
        "knowledge_base": {
            "version": "1.0",
            "total_entries": len(merged_entries),
            "total_sources": len(sources),
            "disciplines": list({
                s.get("discipline", "recurve") for s in sources
            }),
        },
        "sources": sources,
        "entries": list(merged_entries.values()),
    }

    # Stats breakdown
    phases = {}
    categories = {}
    for e in kb["entries"]:
        p = e.get("phase", "unknown")
        c = e.get("category", "unknown")
        phases[p] = phases.get(p, 0) + 1
        categories[c] = categories.get(c, 0) + 1

    kb["knowledge_base"]["entries_by_phase"] = phases
    kb["knowledge_base"]["entries_by_category"] = categories

    out_path = json_dir / "knowledge_base.json"
    out_path.write_text(json.dumps(kb, indent=2, ensure_ascii=False))

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Merged knowledge base")
    log.info("  Total entries : %d", len(merged_entries))
    log.info("  Sources       : %d", len(sources))
    log.info("  By phase      : %s", phases)
    log.info("  By category   : %s", categories)
    log.info("  Output        : %s", out_path)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return kb


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract archery coaching knowledge from YouTube via Gemini",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", type=str, default=None,
                   help="Single YouTube URL to process")
    p.add_argument("--url-file", type=Path, default=None,
                   help="Text file with one YouTube URL per line")
    p.add_argument("--output", type=Path, default=Path("data/knowledge_json"),
                   help="Output directory for JSON files")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip extraction, only merge existing JSON files")
    p.add_argument("--gemini-model", type=str, default="gemini-2.5-flash",
                   help="Gemini model name (e.g. gemini-2.5-flash, gemini-2.5-pro)")
    p.add_argument("--retries", type=int, default=2,
                   help="Number of retry attempts per video on failure")
    p.add_argument("--delay", type=float, default=3.0,
                   help="Seconds to wait between API calls (rate limiting)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    args.output.mkdir(parents=True, exist_ok=True)

    # Merge-only mode
    if args.merge_only:
        merge_knowledge_base(args.output)
        return

    # Collect URLs
    urls = []
    if args.url:
        urls.append(args.url.strip())
    if args.url_file:
        if not args.url_file.exists():
            log.error("URL file not found: %s", args.url_file)
            sys.exit(1)
        for line in args.url_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        log.error("No URLs provided. Use --url or --url-file")
        sys.exit(1)

    # API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error(
            "GEMINI_API_KEY not set. Add it to .env or export it as an env var."
        )
        sys.exit(1)

    client = make_client(api_key)
    log.info("Gemini model: %s", args.gemini_model)
    log.info("Processing %d video(s) → %s", len(urls), args.output)

    succeeded = 0
    failed = []

    for i, url in enumerate(urls):
        log.info("[%d/%d] %s", i + 1, len(urls), url)
        data = extract_from_url(client, url,
                                model_name=args.gemini_model,
                                retries=args.retries)

        if data:
            save_entry_file(data, args.output)
            succeeded += 1
        else:
            failed.append(url)

        # Rate limit pause between calls (not needed after last)
        if i < len(urls) - 1:
            log.debug("Waiting %.1fs before next call…", args.delay)
            time.sleep(args.delay)

    log.info("Extraction complete — %d succeeded, %d failed", succeeded, len(failed))
    if failed:
        log.warning("Failed URLs:")
        for u in failed:
            log.warning("  %s", u)

    # Auto-merge after extraction
    if succeeded > 0:
        log.info("Merging into knowledge_base.json…")
        merge_knowledge_base(args.output)


if __name__ == "__main__":
    main()
