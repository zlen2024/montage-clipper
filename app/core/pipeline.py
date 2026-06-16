"""Job orchestrator: detect -> (optional AI refine) -> render, driving job status.

This is the unit of work scheduled per upload. Keeping it a single function with a
clean signature is what makes the documented Redis/RQ upgrade a drop-in later.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import Settings
from app.core.jobs import Job, JobStatus, JobStore, Tier
from app.core.segment import Segment
from app.detect import audio_loudness, scene_scoring
from app.llm.client_scorer import ClientSubmittedScorer
from app.llm.interface import SceneScorer
from app.media import ffmpeg_io
from app.media.probe import has_audio_stream


def run(
    job_id: str,
    source: Path,
    store: JobStore,
    settings: Settings,
    *,
    override_scorer: SceneScorer | None = None,
) -> None:
    """Process a job end to end. Never raises — failures are recorded on the job.

    `override_scorer` lets tests inject a deterministic scorer and skip the
    browser-scoring wait entirely.
    """
    job = store.get(job_id)
    if job is None:
        return

    work = settings.work_dir / job_id
    try:
        store.set_status(job_id, JobStatus.EXTRACTING, 10)
        if not has_audio_stream(source):
            raise RuntimeError(
                "This video has no audio track. Highlight detection listens for the "
                "loudest moments, so it needs a recording with game or mic audio."
            )
        candidates = audio_loudness.detect(source, work, settings)
        store.set_status(job_id, JobStatus.DETECTING, 40)
        if not candidates:
            raise RuntimeError(
                "No highlights detected — the audio may be too quiet or uniform."
            )

        if job.tier == Tier.AI:
            segments = _refine_with_ai(job, source, candidates, work, store, settings, override_scorer)
        else:
            segments = audio_loudness.select_and_cap(candidates, settings)

        if not segments:
            raise RuntimeError("No segments survived selection.")

        store.set_status(job_id, JobStatus.RENDERING, 80)
        out_path = settings.outputs_dir / job_id / "montage.mp4"
        ffmpeg_io.render_montage(source, segments, work, out_path)

        _cleanup_work(work)
        store.set_status(job_id, JobStatus.DONE, 100)
    except Exception as exc:  # noqa: BLE001 — record any failure on the job
        store.fail(job_id, str(exc))
        _cleanup_work(work)


def _refine_with_ai(
    job: Job,
    source: Path,
    candidates: list[Segment],
    work: Path,
    store: JobStore,
    settings: Settings,
    override_scorer: SceneScorer | None,
) -> list[Segment]:
    """Sample frames, obtain scores (browser or injected), re-rank, and cap."""
    frames_per_segment = _sample_all_frames(job, source, candidates, work, settings)

    if override_scorer is not None:
        scorer: SceneScorer = override_scorer
    else:
        # Hand off to the browser: frames are on disk, expose them and wait.
        store.set_status(job.id, JobStatus.AWAITING_CLIENT_SCORING, 60)
        scores = store.wait_for_scores(job.id, settings.client_scoring_timeout)
        current = store.get(job.id)
        if current is not None and current.status == JobStatus.FAILED:
            return []
        if not scores:
            # Timed out or no scores submitted — degrade to audio-only ranking.
            return audio_loudness.select_and_cap(candidates, settings)
        scorer = ClientSubmittedScorer(scores)

    refined = scene_scoring.refine(candidates, frames_per_segment, scorer, job.id, settings)
    return audio_loudness.select_and_cap(refined, settings)


def _sample_all_frames(
    job: Job, source: Path, candidates: list[Segment], work: Path, settings: Settings
) -> list[list[bytes]]:
    """Sample frames for every candidate; record counts so the API can build URLs."""
    frames_per_segment: list[list[bytes]] = []
    frame_counts: list[int] = []
    for i, seg in enumerate(candidates):
        paths = ffmpeg_io.sample_frames(
            source,
            work / f"seg_{i:03d}",
            seg.start,
            seg.duration,
            settings.frame_fps,
            settings.frame_scale_width,
            settings.max_frames_per_segment,
        )
        frame_counts.append(len(paths))
        frames_per_segment.append([p.read_bytes() for p in paths])

    job.segments = candidates
    job.frame_counts = frame_counts
    return frames_per_segment


def _cleanup_work(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)
