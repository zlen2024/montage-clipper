"""The `SceneScorer` seam.

Everything that turns a segment's frames into an "epicness" score implements this
ABC. The pipeline depends only on the interface, so the actual scorer can be:

  * `ClientSubmittedScorer` — scores computed in the user's browser via WebGPU (MVP),
  * `FakeScorer` — a deterministic stub for tests / offline dev,
  * a future cloud scorer (e.g. Claude vision) — same interface, no pipeline change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.segment import Segment


@dataclass
class SegmentContext:
    """Identifies a segment within a job when scoring."""

    job_id: str
    index: int
    segment: Segment


@dataclass
class SceneScore:
    """Result of scoring one segment."""

    epicness: float          # 0.0 .. 1.0
    reason: str = ""


class SceneScorer(ABC):
    @abstractmethod
    def score_segment(self, frames: list[bytes], context: SegmentContext) -> SceneScore:
        """Return an epicness score for a segment given its sampled frame images."""
        raise NotImplementedError
