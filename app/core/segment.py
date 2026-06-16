"""The `Segment` type — the shared currency between detection, scoring, and render.

Both the free (audio) tier and the paid (AI) tier produce and consume `Segment`s,
so the render path never needs to know which tier created them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """A candidate highlight slice of the source video.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        score: Ranking score. Initially the (normalized) audio loudness; after AI
            re-ranking it becomes the blended audio+epicness score.
        peak: Original loudness at the detected peak (kept so the AI tier can blend
            with epicness even after `score` is overwritten).
    """

    start: float
    end: float
    score: float
    peak: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def clamped(self, lo: float, hi: float) -> "Segment":
        """Return a copy clamped to the [lo, hi] time bounds."""
        return Segment(
            start=max(lo, min(self.start, hi)),
            end=max(lo, min(self.end, hi)),
            score=self.score,
            peak=self.peak,
        )
