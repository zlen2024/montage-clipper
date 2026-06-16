"""Paid-tier refinement: re-rank audio candidates by AI epicness.

This runs *on top of* the audio candidates — it never scans the whole video. Each
candidate's sampled frames are scored by a `SceneScorer` (the browser via WebGPU in
production, a stub in tests). We blend the AI epicness with the audio loudness and
drop "loud but boring" segments (mic bumps, menu noise) before the shared
merge/cap/render path runs.
"""
from __future__ import annotations

from app.config import Settings
from app.core.segment import Segment
from app.llm.interface import SceneScorer, SegmentContext


def refine(
    candidates: list[Segment],
    frames_per_segment: list[list[bytes]],
    scorer: SceneScorer,
    job_id: str,
    settings: Settings,
) -> list[Segment]:
    """Re-score candidates with `scorer` and filter low-epicness ones.

    A segment with no AI score (scorer returned ``None`` — e.g. the browser failed
    on it) keeps its audio loudness and is never filtered out, so the job degrades
    gracefully to audio-only ranking instead of dropping content.
    """
    refined: list[Segment] = []
    for i, seg in enumerate(candidates):
        frames = frames_per_segment[i] if i < len(frames_per_segment) else []
        result = scorer.score_segment(frames, SegmentContext(job_id, i, seg))

        if result is None:
            # No AI signal for this segment: keep audio ranking, don't filter.
            seg.score = seg.peak
            refined.append(seg)
            continue

        if result.epicness < settings.min_epicness:
            continue  # loud but boring — drop it

        seg.score = settings.weight_audio * seg.peak + settings.weight_ai * result.epicness
        refined.append(seg)

    # If filtering removed everything (e.g. every score was below threshold), fall
    # back to the unfiltered audio candidates so the user still gets a montage.
    if not refined:
        for seg in candidates:
            seg.score = seg.peak
        return list(candidates)

    return refined
