#!/usr/bin/env python3
"""Extract frames from gameplay video(s) for labeling.

Turns a video (or a folder of videos) into JPEG frames at a fixed rate, named
``<video-stem>_t<seconds>.jpg`` so you can trace any frame back to its moment.

Examples
--------
    # one video, 1 frame per second
    python extract_frames.py clip.mp4 --out frames/ --fps 1

    # every video in a folder, 1 frame every 2 seconds, downscaled
    python extract_frames.py videos/ --out frames/ --fps 0.5 --width 512

Then sort the JPEGs in ``frames/`` into ``dataset/kill/`` and ``dataset/no_kill/``.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        sys.exit("ffmpeg not found on PATH. Install it: https://ffmpeg.org/download.html")
    return path


def extract_one(video: Path, out_dir: Path, fps: float, width: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # %05d here is a frame counter; we rename to timestamps afterward.
    pattern = out_dir / f"{video.stem}_%05d.jpg"
    cmd = [
        _ffmpeg(), "-y",
        "-i", str(video),
        "-vf", f"fps={fps},scale={width}:-1",
        "-q:v", "4",
        str(pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! ffmpeg failed for {video.name}:\n{proc.stderr[-600:]}", file=sys.stderr)
        return 0

    # Rename frame_00001 -> stem_t0.0, stem_t2.0, ... using the known fps.
    frames = sorted(out_dir.glob(f"{video.stem}_[0-9]*.jpg"))
    period = 1.0 / fps
    for i, f in enumerate(frames):
        ts = i * period
        f.rename(out_dir / f"{video.stem}_t{ts:.1f}.jpg")
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="A video file or a folder of videos")
    ap.add_argument("--out", type=Path, required=True, help="Output folder for frames")
    ap.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default 1)")
    ap.add_argument("--width", type=int, default=512, help="Downscale width in px (default 512)")
    args = ap.parse_args()

    if args.input.is_dir():
        videos = [p for p in sorted(args.input.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    else:
        videos = [args.input]
    if not videos:
        sys.exit(f"No videos found in {args.input}")

    total = 0
    for v in videos:
        print(f"Extracting {v.name} …")
        total += extract_one(v, args.out, args.fps, args.width)
    print(f"\nDone — {total} frames in {args.out}")
    print("Next: sort them into dataset/kill/ and dataset/no_kill/ (see training/data/README.md)")


if __name__ == "__main__":
    main()
