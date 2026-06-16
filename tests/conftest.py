"""Shared test fixtures: a temp-storage Settings and a synthetic gameplay clip.

The synthetic clip has loud audio bursts at known timestamps (5s, 20s, 40s), so
detection can be asserted exactly without a real gameplay recording.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.media.probe import ffmpeg_available

BURST_TIMES = (5.0, 20.0, 40.0)
SAMPLE_DURATION = 45.0
SAMPLE_RATE = 16000


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(storage_dir=tmp_path / "storage")
    s.ensure_dirs()
    return s


def _write_burst_wav(path: Path) -> None:
    t = np.linspace(0, SAMPLE_DURATION, int(SAMPLE_DURATION * SAMPLE_RATE), endpoint=False)
    audio = 0.02 * np.sin(2 * np.pi * 220 * t)  # quiet base tone
    for c in BURST_TIMES:
        mask = (t >= c) & (t < c + 0.8)
        audio[mask] += 0.6 * np.sin(2 * np.pi * 880 * t[mask])  # loud burst
    sf.write(str(path), audio.astype(np.float32), SAMPLE_RATE)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")

    wav = tmp_path / "audio.wav"
    _write_burst_wav(wav)

    out = tmp_path / "sample.mp4"
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=navy:s=320x180:d={SAMPLE_DURATION}",
            "-i", str(wav),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"Failed to build sample clip:\n{proc.stderr[-1500:]}")
    return out
