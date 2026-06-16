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


class FrameSegment(BaseModel):
    index: int
    start: float
    end: float
    frame_urls: list[str]


class FramesManifest(BaseModel):
    job_id: str
    model_id: str = Field(description="Hugging Face model id the browser should load")
    segments: list[FrameSegment]


class ScoreSubmission(BaseModel):
    # Keys are segment indices as strings (JSON object keys); values are epicness 0..1.
    scores: dict[str, float]
