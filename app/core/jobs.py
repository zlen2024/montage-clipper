"""In-memory job store with a JSON sidecar, plus the client-scoring handoff.

For the MVP this is a process-local dict; each job also writes a `job.json` sidecar
so its user-facing status survives a restart. The store is deliberately small and
hidden behind a handful of methods so it can later be swapped for Redis without the
pipeline noticing (see the plan's upgrade path).

The AI tier needs the server pipeline to *pause* until the browser posts scores;
that handoff is a per-job `threading.Event` (jobs run in FastAPI's threadpool, so a
blocking wait is fine).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.core.segment import Segment


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    DETECTING = "detecting"
    AWAITING_CLIENT_SCORING = "awaiting_client_scoring"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


class Tier(str, Enum):
    FREE = "free"
    AI = "ai"


@dataclass
class Job:
    id: str
    tier: Tier
    filename: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    # Runtime-only (not persisted): present while an AI job awaits browser scores.
    segments: list[Segment] = field(default_factory=list)
    frame_counts: list[int] = field(default_factory=list)  # legacy: per-segment frame count
    ai_frame_times: list[float] = field(default_factory=list)  # AI-first: timestamp per frame

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "filename": self.filename,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "output_ready": self.status == JobStatus.DONE,
        }


class JobStore:
    def __init__(self, outputs_dir: Path):
        self._jobs: dict[str, Job] = {}
        self._scores: dict[str, dict[int, float]] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._outputs_dir = outputs_dir

    # --- lifecycle ---
    def create(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._events[job.id] = threading.Event()
        self._persist(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatus, progress: int | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = status
        if progress is not None:
            job.progress = progress
        self._persist(job)

    def set_progress(self, job_id: str, progress: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.progress = max(0, min(100, progress))
        self._persist(job)

    def fail(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.FAILED
        job.error = error
        self._persist(job)
        # Unblock any pipeline waiting on scores so it can observe the failure.
        ev = self._events.get(job_id)
        if ev is not None:
            ev.set()

    # --- client scoring handoff ---
    def submit_scores(self, job_id: str, scores: dict[int, float]) -> bool:
        if job_id not in self._jobs:
            return False
        with self._lock:
            self._scores[job_id] = {int(k): float(v) for k, v in scores.items()}
            ev = self._events.get(job_id)
        if ev is not None:
            ev.set()
        return True

    def wait_for_scores(self, job_id: str, timeout: float) -> dict[int, float] | None:
        """Block until scores are submitted or `timeout` elapses.

        Returns the score map, or ``None`` on timeout (caller falls back to audio).
        """
        ev = self._events.get(job_id)
        if ev is None:
            return None
        got = ev.wait(timeout=timeout)
        if not got:
            return None
        return self._scores.get(job_id)

    # --- persistence ---
    def _persist(self, job: Job) -> None:
        path = self._outputs_dir / job.id / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job.public_dict(), indent=2), encoding="utf-8")
