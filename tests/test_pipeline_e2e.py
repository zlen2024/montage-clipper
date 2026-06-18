"""End-to-end pipeline test against a synthetic clip (skips without ffmpeg)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.core.jobs import Job, JobStatus, JobStore, Tier
from app.core.pipeline import run as run_pipeline
from app.llm.interface import SceneScore, SceneScorer, SegmentContext
from app.media.probe import ffmpeg_available, probe_duration


class _ScripedScorer(SceneScorer):
    """Scores YES for every even-indexed frame, NO for odd. Deterministic."""

    def score_segment(self, frames, context: SegmentContext) -> SceneScore | None:
        return SceneScore(epicness=0.9 if context.index % 2 == 0 else 0.1, reason="stub")


def _run(settings: Settings, source: Path, tier: Tier, **kwargs) -> Job:
    store = JobStore(settings.outputs_dir)
    store.create(Job(id="testjob", tier=tier, filename=source.name))
    run_pipeline("testjob", source, store, settings, **kwargs)
    return store.get("testjob")


def test_free_tier_end_to_end(settings: Settings, sample_video: Path):
    job = _run(settings, sample_video, Tier.FREE)

    assert job.status == JobStatus.DONE, job.error
    montage = settings.outputs_dir / "testjob" / "montage.mp4"
    assert montage.exists() and montage.stat().st_size > 0

    duration = probe_duration(montage)
    assert 0 < duration <= settings.max_montage_len + 1.0

    # Work dir cleaned up after a successful render.
    assert not (settings.work_dir / "testjob").exists()


def test_ai_tier_end_to_end_with_stub(settings: Settings, sample_video: Path):
    # The stub scores half the sampled frames as YES — the AI-first flow should
    # turn each YES into a clip and produce a montage.
    job = _run(settings, sample_video, Tier.AI, override_scorer=_ScripedScorer())

    assert job.status == JobStatus.DONE, job.error
    montage = settings.outputs_dir / "testjob" / "montage.mp4"
    assert montage.exists() and montage.stat().st_size > 0
    assert probe_duration(montage) > 0


def test_video_without_audio_fails_clearly(settings: Settings, tmp_path: Path):
    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")

    # A video-only clip (no audio stream), like many screen recordings.
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
        capture_output=True, text=True, check=True,
    )

    job = _run(settings, silent, Tier.FREE)
    assert job.status == JobStatus.FAILED
    assert "no audio" in (job.error or "").lower()
