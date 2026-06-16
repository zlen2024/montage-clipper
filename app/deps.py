"""Lazily-built shared singletons (settings, job store, credit service).

Exposed as `lru_cache`d functions so FastAPI's `Depends` can inject them and tests
can reset them (clear the caches after pointing `STORAGE_DIR` at a temp dir).
"""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.core.credits import CreditService
from app.core.jobs import JobStore


@lru_cache
def get_store() -> JobStore:
    settings: Settings = get_settings()
    settings.ensure_dirs()
    return JobStore(settings.outputs_dir)


@lru_cache
def get_credits() -> CreditService:
    settings: Settings = get_settings()
    settings.ensure_dirs()
    return CreditService(settings.storage_dir / "accounts.json", settings.free_daily_limit)
