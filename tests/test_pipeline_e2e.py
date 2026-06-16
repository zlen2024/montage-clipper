"""End-to-end pipeline test against a synthetic clip (skips without ffmpeg)."""
from __future__ import annotations

from pathlib import Path


from app.config import Settings
from app.core.jobs import Job, JobStatus, JobStore, Tier
from app.core.pipeline import run as run_pipeline
from app.llm.fake_scorer import FakeScorer
from app.media.probe import probe_duration


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
    # FakeScorer skips the browser wait and scores frames deterministically.
    job = _run(settings, sample_video, Tier.AI, override_scorer=FakeScorer())

    assert job.status == JobStatus.DONE, job.error
    montage = settings.outputs_dir / "testjob" / "montage.mp4"
    assert montage.exists() and montage.stat().st_size > 0
    assert probe_duration(montage) > 0
