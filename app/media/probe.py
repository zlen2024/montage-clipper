"""Thin ffprobe wrappers for inspecting media files."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegNotInstalled(RuntimeError):
    """Raised when the ffmpeg/ffprobe binaries are not on PATH."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise FFmpegNotInstalled("ffprobe is not installed or not on PATH")
    return path


def probe_duration(source: Path) -> float:
    """Return the container duration in seconds, or 0.0 if unknown."""
    ffprobe = _require_ffprobe()
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout or "{}")
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return 0.0
