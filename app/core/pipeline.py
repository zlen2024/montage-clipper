"""Job orchestrator: detect -> render, driving job status.

Two algorithms behind one entry point:

* **Free tier** — audio-loudness detection only. The loudest moments become
  highlights.
* **AI tier** — the VLM is the *primary* decision-maker. We sample the whole
  video at a fixed cadence and ask the model, frame by frame, whether there is
  a kill / active combat. Each YES becomes a highlight.

Keeping this a single function with a clean signature is what makes the
documented Redis/RQ upgrade a drop-in later.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import Settings
from app.core.jobs import Job, JobStatus, JobStore, Tier
from app.core.segment import Segment
from app.detect import ai_first, audio_loudness
from app.llm.interface import SceneScorer, SegmentContext
from app.media import ffmpeg_io
from app.media.probe import has_audio_stream, probe_duration


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
        if job.tier == Tier.AI:
            segments = _run_ai_first(job, source, work, store, settings, override_scorer)
        else:
            segments = _run_free(source, work, settings)

        if not segments:
            raise RuntimeError("No highlights found in this video.")

        store.set_status(job_id, JobStatus.RENDERING, 80)
        out_path = settings.outputs_dir / job_id / "montage.mp4"
        ffmpeg_io.render_montage(source, segments, work, out_path)

        _cleanup_work(work)
        store.set_status(job_id, JobStatus.DONE, 100)
    except Exception as exc:  # noqa: BLE001 — record any failure on the job
        store.fail(job_id, str(exc))
        _cleanup_work(work)


def _run_free(source: Path, work: Path, settings: Settings) -> list[Segment]:
    """Audio-loudness algorithm: extract audio, peak-pick, merge, cap."""
    if not has_audio_stream(source):
        raise RuntimeError(
            "This video has no audio track. The free mode finds highlights by "
            "listening for the loudest moments — try the AI mode instead, or "
            "use a recording with game/mic audio."
        )
    candidates = audio_loudness.detect(source, work, settings)
    if not candidates:
        raise RuntimeError(
            "No loud moments detected — the audio may be too quiet or uniform. "
            "Try AI mode for a content-based pass."
        )
    return audio_loudness.select_and_cap(candidates, settings)


def _run_ai_first(
    job: Job,
    source: Path,
    work: Path,
    store: JobStore,
    settings: Settings,
    override_scorer: SceneScorer | None,
) -> list[Segment]:
    """Ask the VLM about every Nth second of the video; clip the YES moments."""
    duration = probe_duration(source)
    if duration <= 0:
        raise RuntimeError("Could not read the video's duration.")

    # 1. Sample the whole video at a fixed cadence.
    store.set_status(job.id, JobStatus.DETECTING, 20)
    samples, _ = ai_first.sample_uniform(source, work, duration, settings)
    if not samples:
        raise RuntimeError("Could not sample frames from this video.")
    job.ai_frame_times = [t for t, _ in samples]

    # 2. Get a score per frame (browser-side, or test stub).
    if override_scorer is not None:
        scores = _score_with_stub(override_scorer, job.id, samples)
    else:
        store.set_status(job.id, JobStatus.AWAITING_CLIENT_SCORING, 35)
        client_scores = store.wait_for_scores(job.id, settings.client_scoring_timeout)
        current = store.get(job.id)
        if current is not None and current.status == JobStatus.FAILED:
            return []
        if not client_scores:
            raise RuntimeError(
                "AI scoring didn't finish in time. Try a shorter clip, or use Free mode."
            )
        scores = client_scores

    # 3. Build a clip around each YES frame, then merge + cap.
    store.set_status(job.id, JobStatus.RENDERING, 70)
    raw_segs = ai_first.scores_to_segments(samples, scores, settings, duration)
    if not raw_segs:
        raise RuntimeError(
            "The AI didn't find any kill / combat moments. Try a different clip, "
            "lower AI_KILL_THRESHOLD, or use Free mode."
        )
    merged = audio_loudness.merge_segments(raw_segs, settings)
    return audio_loudness.select_and_cap(merged, settings)


def _score_with_stub(
    scorer: SceneScorer,
    job_id: str,
    samples: list[tuple[float, Path]],
) -> dict[int, float]:
    """Run an injected scorer over each sampled frame (used by tests)."""
    out: dict[int, float] = {}
    for i, (t, path) in enumerate(samples):
        ctx = SegmentContext(job_id, i, Segment(t, t, 0.0, 0.0))
        result = scorer.score_segment([path.read_bytes()], ctx)
        if result is not None:
            out[i] = result.epicness
    return out


def _cleanup_work(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)
