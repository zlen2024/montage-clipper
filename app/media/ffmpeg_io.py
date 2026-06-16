"""All ffmpeg calls live here — extract audio, sample frames, trim, and concat.

We shell out to the ffmpeg CLI with explicit argument lists (rather than a wrapper
library) so the exact command is transparent and debuggable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.core.segment import Segment
from app.media.probe import FFmpegNotInstalled


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotInstalled("ffmpeg is not installed or not on PATH")
    return path


def _run(args: list[str]) -> None:
    """Run ffmpeg, raising with stderr captured on failure."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): {' '.join(args)}\n{proc.stderr[-2000:]}"
        )


def extract_audio(source: Path, out_wav: Path, sample_rate: int) -> Path:
    """Downmix to mono and resample to `sample_rate`, writing a WAV file."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _ffmpeg(), "-y",
        "-i", str(source),
        "-vn",                  # drop video
        "-ac", "1",             # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        str(out_wav),
    ])
    return out_wav


def sample_frames(
    source: Path,
    out_dir: Path,
    start: float,
    duration: float,
    fps: float,
    scale_width: int,
    max_frames: int,
) -> list[Path]:
    """Sample frames from a segment as JPEGs and return the written paths (ordered)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%03d.jpg"
    _run([
        _ffmpeg(), "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", f"fps={fps},scale={scale_width}:-1",
        "-frames:v", str(max_frames),
        "-q:v", "5",
        str(pattern),
    ])
    return sorted(out_dir.glob("frame_*.jpg"))


def cut_clip(source: Path, out_clip: Path, start: float, duration: float) -> Path:
    """Cut a single clip, re-encoding for clean keyframe-aligned boundaries."""
    out_clip.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _ffmpeg(), "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(out_clip),
    ])
    return out_clip


def render_montage(source: Path, segments: list[Segment], work_dir: Path, out_path: Path) -> Path:
    """Cut each segment then concat them losslessly into the final montage.

    Segments are cut (re-encoded) individually so the concat demuxer can stream-copy
    them. The caller is responsible for ordering segments chronologically.
    """
    if not segments:
        raise ValueError("render_montage requires at least one segment")

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for i, seg in enumerate(segments):
        clip = cut_clip(source, work_dir / f"clip_{i:03d}.mp4", seg.start, seg.duration)
        clips.append(clip)

    # Concat demuxer needs a list file with one `file '<path>'` per line.
    list_file = work_dir / "concat_list.txt"
    list_file.write_text(
        "".join(f"file '{clip.resolve().as_posix()}'\n" for clip in clips),
        encoding="utf-8",
    )

    _run([
        _ffmpeg(), "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ])
    return out_path
