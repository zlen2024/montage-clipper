"""Free-tier highlight detection from audio loudness.

The action in gameplay (gunfire, explosions, the player reacting) is almost always
the loudest part of the recording. We compute a smoothed energy envelope, pick the
most prominent peaks, pad them into segments, merge overlaps, and cap the total
length — producing the same `Segment` list the AI tier later refines.

The signal-processing functions take plain numpy arrays so they can be unit-tested
with synthetic envelopes; only `detect` and `detect_from_wav` touch files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from app.config import Settings
from app.core.segment import Segment
from app.media.ffmpeg_io import extract_audio


def compute_envelope(samples: np.ndarray, sr: int, settings: Settings) -> tuple[np.ndarray, float]:
    """Return a smoothed loudness envelope in dB and the hop interval (seconds).

    The envelope is per-frame RMS converted to dB, then smoothed with a moving
    average so a single transient (one gunshot) doesn't fragment into many peaks.
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    win_n = max(1, int(settings.win_seconds * sr))
    hop_n = max(1, int(settings.hop_seconds * sr))

    if samples.size < win_n:
        return np.array([]), settings.hop_seconds

    n_frames = 1 + (samples.size - win_n) // hop_n
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = samples[i * hop_n : i * hop_n + win_n]
        rms[i] = np.sqrt(np.mean(frame * frame))

    env_db = 20.0 * np.log10(rms + 1e-9)

    smooth_n = max(1, int(settings.smooth_seconds / settings.hop_seconds))
    env_smoothed = uniform_filter1d(env_db, size=smooth_n, mode="nearest")
    return env_smoothed, settings.hop_seconds


def pick_peaks(env: np.ndarray, hop: float, settings: Settings) -> list[tuple[float, float]]:
    """Pick prominent loudness peaks. Returns (time_seconds, height_db) pairs.

    A dynamic threshold (mean + k·std) adapts to each clip's loudness, `distance`
    enforces minimum spacing, and `prominence` rejects plateaus.
    """
    if env.size == 0:
        return []

    mean = float(np.mean(env))
    std = float(np.std(env))
    height = mean + settings.threshold_k * std
    distance = max(1, int(settings.min_peak_gap / hop))
    prominence = max(0.5 * std, 1e-6)

    indices, props = find_peaks(env, height=height, distance=distance, prominence=prominence)
    heights = props.get("peak_heights", env[indices] if indices.size else np.array([]))
    return [(int(idx) * hop, float(h)) for idx, h in zip(indices, heights)]


def peaks_to_segments(
    peaks: list[tuple[float, float]], duration: float, settings: Settings
) -> list[Segment]:
    """Pad each peak into a [t - lead_in, t + lead_out] window, clamped to the clip.

    `peak` is normalized 0..1 across the candidate set so the AI tier can blend it
    directly with epicness.
    """
    if not peaks:
        return []

    heights = np.array([h for _, h in peaks], dtype=np.float64)
    lo, hi = float(heights.min()), float(heights.max())
    span = hi - lo if hi > lo else 1.0

    segments: list[Segment] = []
    for t, h in peaks:
        norm = (h - lo) / span
        seg = Segment(start=t - settings.lead_in, end=t + settings.lead_out, score=norm, peak=norm)
        segments.append(seg.clamped(0.0, duration))
    return segments


def merge_segments(segments: list[Segment], settings: Settings) -> list[Segment]:
    """Merge overlapping or near-adjacent segments (within `merge_gap`)."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    merged: list[Segment] = [
        Segment(ordered[0].start, ordered[0].end, ordered[0].score, ordered[0].peak)
    ]
    for seg in ordered[1:]:
        cur = merged[-1]
        if seg.start <= cur.end + settings.merge_gap:
            cur.end = max(cur.end, seg.end)
            cur.peak = max(cur.peak, seg.peak)
            cur.score = max(cur.score, seg.score)
        else:
            merged.append(Segment(seg.start, seg.end, seg.score, seg.peak))
    return merged


def select_and_cap(segments: list[Segment], settings: Settings) -> list[Segment]:
    """Greedily pick the loudest segments up to `max_montage_len`, then order in time.

    The last segment is trimmed to fit the cap exactly (dropped if the remainder is
    too short to be worth a cut).
    """
    if not segments:
        return []

    chosen: list[Segment] = []
    total = 0.0
    for seg in sorted(segments, key=lambda s: s.score, reverse=True):
        remaining = settings.max_montage_len - total
        if remaining <= 0.5:
            break
        if seg.duration <= remaining:
            chosen.append(seg)
            total += seg.duration
        else:
            trimmed = Segment(seg.start, seg.start + remaining, seg.score, seg.peak)
            chosen.append(trimmed)
            total += trimmed.duration
            break

    chosen.sort(key=lambda s: s.start)
    return chosen


def detect_candidates(samples: np.ndarray, sr: int, duration: float, settings: Settings) -> list[Segment]:
    """Full audio pipeline up to the *candidate* set (merged, not yet capped).

    Returns more segments than the final montage needs (`candidate_overshoot`) so the
    AI tier has room to re-rank. Call `select_and_cap` to finalize for the free tier.
    """
    env, hop = compute_envelope(samples, sr, settings)
    peaks = pick_peaks(env, hop, settings)

    # Keep the loudest candidates; overshoot the target so AI re-ranking has options.
    keep = max(settings.target_segments * settings.candidate_overshoot, settings.target_segments)
    peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:keep]

    segments = peaks_to_segments(peaks, duration, settings)
    return merge_segments(segments, settings)


def detect_from_wav(wav_path: Path, settings: Settings) -> list[Segment]:
    """Load an extracted WAV and produce candidate segments."""
    samples, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if samples.ndim > 1:  # safety: downmix if somehow stereo
        samples = samples.mean(axis=1)
    duration = samples.shape[0] / float(sr) if sr else 0.0
    return detect_candidates(samples, sr, duration, settings)


def detect(source: Path, work_dir: Path, settings: Settings) -> list[Segment]:
    """Extract audio from `source` and return candidate segments."""
    wav = extract_audio(source, work_dir / "audio.wav", settings.sample_rate)
    return detect_from_wav(wav, settings)
