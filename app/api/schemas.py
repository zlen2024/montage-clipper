"""Pydantic request/response models for the job API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    status: str
    tier: str


class JobStatusResponse(BaseModel):
    id: str
    tier: str
    filename: str
    status: str
    progress: int
    error: str | None = None
    created_at: float
    output_ready: bool


class AIFrame(BaseModel):
    """One frame the browser must score (AI-first tier)."""

    index: int
    timestamp: float  # source video time in seconds
    url: str


class FramesManifest(BaseModel):
    job_id: str
    model_id: str = Field(description="Hugging Face model id the browser should load")
    frames: list[AIFrame]


class ScoreSubmission(BaseModel):
    # Keys are frame indices as strings (JSON object keys); values are 0..1.
    scores: dict[str, float]
