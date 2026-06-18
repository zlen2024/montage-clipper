"""AI-first highlight detection: ask a VLM about every Nth second of video.

Instead of using audio loudness to pick candidates and refining them with a
model, the AI tier samples the whole video at a fixed cadence and asks the VLM
directly: "is there a kill / active combat here?" Each YES becomes a montage
clip. This catches silent kills the audio approach would miss and drops loud
non-kill moments (reloads, mic bumps).

The cadence is automatically widened for long videos so we never exceed
`ai_max_frames` — otherwise a 1-hour clip would mean an hour of inference.
"""
from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.segment import Segment
from app.media import ffmpeg_io


def choose_period(duration: float, settings: Settings) -> float:
    """Return the actual sample cadence (widening for long videos)."""
    target = settings.ai_sample_period_s
    n_estimated = int(duration / target) + 1
    if n_estimated > settings.ai_max_frames:
        return duration / settings.ai_max_frames
    return target


def sample_uniform(
    source: Path,
    work_dir: Path,
    duration: float,
    settings: Settings,
) -> tuple[list[tuple[float, Path]], float]:
    """Sample frames at a fixed cadence across the source.

    Returns (samples, period) where `samples` is a list of (timestamp, path) and
    `period` is the cadence actually used (it may be wider than the configured
    default for long videos).
    """
    period = choose_period(duration, settings)
    out_dir = work_dir / "kf"
    paths = ffmpeg_io.sample_at_cadence(
        source,
        out_dir,
        period,
        settings.frame_scale_width,
        settings.ai_max_frames,
    )
    samples = [(i * period, p) for i, p in enumerate(paths)]
    return samples, period


def scores_to_segments(
    samples: list[tuple[float, Path]],
    scores: dict[int, float],
    settings: Settings,
    duration: float,
) -> list[Segment]:
    """Build padded segments from frames the model said YES to.

    Frames the model couldn't score (missing from `scores`) and frames below
    `ai_kill_threshold` are dropped.
    """
    segs: list[Segment] = []
    for i, (t, _) in enumerate(samples):
        score = scores.get(i)
        if score is None or score < settings.ai_kill_threshold:
            continue
        seg = Segment(
            start=t - settings.lead_in,
            end=t + settings.lead_out,
            score=score,
            peak=score,
        )
        segs.append(seg.clamped(0.0, duration))
    return segs
