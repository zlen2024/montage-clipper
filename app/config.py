"""Typed, env-driven application settings.

Everything tunable lives here so the detection algorithm, montage shape, and AI
model can be changed without touching code. Values come from environment
variables or a local `.env` file (see `.env.example`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Storage ---
    storage_dir: Path = Path("storage")
    retention_hours: float = 24.0

    # --- Upload limits ---
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB
    max_duration_seconds: float = 3900.0  # ~65 min, a little headroom over the 1h target

    # --- Montage shape ---
    max_montage_len: float = 60.0  # seconds, hard cap on the output
    target_segments: int = 8       # how many highlights we aim to include

    # --- Audio detection (windowed RMS -> peak picking) ---
    sample_rate: int = 16000       # mono; plenty for energy detection
    win_seconds: float = 0.05      # 50 ms analysis window
    hop_seconds: float = 0.025     # 25 ms hop (50% overlap)
    smooth_seconds: float = 0.5    # envelope smoothing window
    threshold_k: float = 1.0       # peak height = mean + k * std (in dB)
    min_peak_gap: float = 3.0      # minimum spacing between peaks (seconds)
    lead_in: float = 2.0           # padding before each peak
    lead_out: float = 3.0          # padding after each peak (payoff follows the trigger)
    merge_gap: float = 1.0         # merge segments closer than this (seconds)
    candidate_overshoot: int = 3   # keep target_segments * this many candidates

    # --- AI tier (browser WebGPU) ---
    # CLIP zero-shot classifier run client-side; scores "action vs idle" per frame.
    vlm_model_id: str = "Xenova/clip-vit-base-patch32"
    frame_fps: float = 1.0             # frames sampled per second within a segment
    frame_scale_width: int = 512       # downscale long edge to control client cost
    max_frames_per_segment: int = 6
    client_scoring_timeout: float = 300.0  # wait this long for browser scores, else audio-only
    weight_audio: float = 0.3          # blend weight for normalized loudness
    weight_ai: float = 0.7             # blend weight for AI epicness
    min_epicness: float = 0.15         # drop candidates the VLM scores below this

    # --- Free-tier usage cap ---
    free_daily_limit: int = 5

    # --- Dev override ---
    # When true, the AI tier is allowed without credits and the daily free limit
    # is ignored. Use only for local testing — set BYPASS_CREDITS=true.
    bypass_credits: bool = False

    # --- Derived paths ---
    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def work_dir(self) -> Path:
        return self.storage_dir / "work"

    @property
    def outputs_dir(self) -> Path:
        return self.storage_dir / "outputs"

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.work_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
