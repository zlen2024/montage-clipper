"""Unit tests for AI re-ranking (no browser/WebGPU needed)."""
from __future__ import annotations

from app.config import Settings
from app.core.segment import Segment
from app.detect import scene_scoring
from app.llm.client_scorer import ClientSubmittedScorer


def _candidates() -> list[Segment]:
    return [
        Segment(0.0, 5.0, 0.8, 0.8),    # 0: loud, AI says epic
        Segment(20.0, 25.0, 0.7, 0.7),  # 1: loud, AI says boring -> filtered
        Segment(40.0, 45.0, 0.6, 0.6),  # 2: no AI score -> kept on audio
    ]


def test_refine_filters_low_epicness_and_blends():
    settings = Settings()  # min_epicness=0.15, weights 0.3/0.7
    scorer = ClientSubmittedScorer({0: 0.9, 1: 0.05})  # index 2 absent
    refined = scene_scoring.refine(_candidates(), [[], [], []], scorer, "job", settings)

    starts = sorted(s.start for s in refined)
    assert starts == [0.0, 40.0]  # boring segment (index 1) dropped

    epic_seg = next(s for s in refined if s.start == 0.0)
    expected = 0.3 * 0.8 + 0.7 * 0.9
    assert abs(epic_seg.score - expected) < 1e-9

    audio_only_seg = next(s for s in refined if s.start == 40.0)
    assert audio_only_seg.score == 0.6  # fell back to audio loudness


def test_refine_falls_back_when_everything_filtered():
    settings = Settings()
    scorer = ClientSubmittedScorer({0: 0.01, 1: 0.02, 2: 0.0})  # all below threshold
    refined = scene_scoring.refine(_candidates(), [[], [], []], scorer, "job", settings)

    # Rather than returning nothing, it keeps the audio candidates.
    assert len(refined) == 3
    assert all(s.score == s.peak for s in refined)
