"""Unit tests for the audio-loudness signal processing (no ffmpeg needed)."""
from __future__ import annotations

import numpy as np

from app.config import Settings
from app.core.segment import Segment
from app.detect import audio_loudness


def _make_envelope(duration_s: float, hop: float, peak_times: list[float]) -> np.ndarray:
    """Baseline dB envelope with sharp Gaussian bumps at the given times."""
    n = int(duration_s / hop)
    env = np.full(n, -45.0)
    for t in peak_times:
        center = int(t / hop)
        for i in range(max(0, center - 6), min(n, center + 6)):
            env[i] += 40.0 * np.exp(-((i - center) ** 2) / (2 * 3.0**2))
    return env


def test_pick_peaks_finds_planted_peaks():
    settings = Settings()
    hop = settings.hop_seconds
    planted = [5.0, 20.0, 40.0]
    env = _make_envelope(45.0, hop, planted)

    peaks = audio_loudness.pick_peaks(env, hop, settings)
    found = sorted(t for t, _ in peaks)

    assert len(found) == len(planted)
    for got, want in zip(found, planted):
        assert abs(got - want) < 0.2  # within ~8 frames


def test_merge_segments_combines_overlapping():
    settings = Settings()
    segs = [
        Segment(0.0, 5.0, 0.8, 0.8),
        Segment(4.5, 9.0, 0.9, 0.9),   # overlaps the first
        Segment(20.0, 23.0, 0.5, 0.5),  # separate
    ]
    merged = audio_loudness.merge_segments(segs, settings)

    assert len(merged) == 2
    assert merged[0].start == 0.0 and merged[0].end == 9.0
    assert merged[0].peak == 0.9  # keeps the louder peak


def test_select_and_cap_respects_length_and_order():
    settings = Settings(max_montage_len=10.0)
    segs = [
        Segment(40.0, 45.0, 0.3, 0.3),  # 5s, quietest
        Segment(0.0, 5.0, 0.9, 0.9),    # 5s, loudest
        Segment(20.0, 25.0, 0.6, 0.6),  # 5s, middle
    ]
    chosen = audio_loudness.select_and_cap(segs, settings)

    total = sum(s.duration for s in chosen)
    assert total <= settings.max_montage_len + 1e-6
    # Output ordered chronologically...
    assert [s.start for s in chosen] == sorted(s.start for s in chosen)
    # ...and built from the loudest segments first (quietest dropped at the 10s cap).
    assert all(s.start != 40.0 for s in chosen)


def test_peaks_to_segments_clamps_to_bounds():
    settings = Settings(lead_in=2.0, lead_out=3.0)
    segs = audio_loudness.peaks_to_segments([(0.5, 10.0), (44.5, 12.0)], duration=45.0, settings=settings)

    for s in segs:
        assert s.start >= 0.0
        assert s.end <= 45.0
