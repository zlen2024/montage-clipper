"""TTL cleanup: remove job artifacts older than the retention window.

Keeps local disk bounded for the single-box MVP. Operates purely on directory
modification time, so it works even after a restart wiped the in-memory store.
"""
from __future__ import annotations

import shutil
import time

from app.config import Settings


def sweep(settings: Settings) -> int:
    """Delete per-job directories older than `retention_hours`. Returns count removed."""
    cutoff = time.time() - settings.retention_hours * 3600.0
    removed = 0
    for base in (settings.uploads_dir, settings.work_dir, settings.outputs_dir):
        if not base.exists():
            continue
        for job_dir in base.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                if job_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    return removed
