"""Deterministic scorer for tests and offline development — zero cost, no network.

Scores from mean pixel brightness of the frames (brighter / busier frames score
higher), with a stable fallback when frames can't be decoded. This lets the whole
pipeline run end-to-end in CI without a browser or WebGPU.
"""
from __future__ import annotations

import io

from app.llm.interface import SceneScore, SceneScorer, SegmentContext


class FakeScorer(SceneScorer):
    def score_segment(self, frames: list[bytes], context: SegmentContext) -> SceneScore:
        if not frames:
            return SceneScore(epicness=0.0, reason="no frames")

        brightness = _mean_brightness(frames[len(frames) // 2])
        if brightness is None:
            # Deterministic fallback keyed on the segment index.
            epicness = ((context.index * 37) % 100) / 100.0
            return SceneScore(epicness=epicness, reason="fallback (undecodable frame)")
        return SceneScore(epicness=brightness, reason=f"brightness={brightness:.2f}")


def _mean_brightness(jpeg_bytes: bytes) -> float | None:
    """Mean luminance in 0..1, or None if Pillow isn't available / decode fails."""
    try:
        from PIL import Image  # optional dependency; not required for the stub to work
    except Exception:
        return None
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            gray = img.convert("L")
            pixels = list(gray.getdata())
        if not pixels:
            return None
        return (sum(pixels) / len(pixels)) / 255.0
    except Exception:
        return None
