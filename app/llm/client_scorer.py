"""Scorer backed by epicness values computed in the user's browser (WebGPU).

The server doesn't run any model here — it's the coordinator. The browser fetches
sampled frames, scores them with SmolVLM via WebGPU, and POSTs back a
``{segment_index: epicness}`` map. This scorer simply surfaces those values to the
pipeline behind the common `SceneScorer` interface.

Segments the browser didn't score (e.g. a partial failure) come back as `None`,
which the pipeline treats as "fall back to audio loudness for this segment".
"""
from __future__ import annotations

from app.llm.interface import SceneScore, SceneScorer, SegmentContext


class ClientSubmittedScorer(SceneScorer):
    def __init__(self, scores_by_index: dict[int, float]):
        # Defensive copy; clamp to [0, 1].
        self._scores = {int(k): max(0.0, min(1.0, float(v))) for k, v in scores_by_index.items()}

    def score_segment(self, frames: list[bytes], context: SegmentContext) -> SceneScore | None:
        if context.index not in self._scores:
            return None  # signals: no client score; pipeline uses audio loudness
        return SceneScore(epicness=self._scores[context.index], reason="webgpu client")
