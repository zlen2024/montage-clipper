"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_jobs import router as jobs_router
from app.config import get_settings
from app.core.cleanup import sweep
from app.media.probe import ffmpeg_available

_STATIC_DIR = Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()

    async def _cleanup_loop() -> None:
        while True:
            sweep(settings)
            await asyncio.sleep(3600)  # hourly

    sweep(settings)  # one sweep at startup
    task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Montage Clipper", version="0.1.0", lifespan=lifespan)
app.include_router(jobs_router)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/healthz")
def healthz() -> JSONResponse:
    ok = ffmpeg_available()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "ffmpeg": ok},
    )


@app.get("/")
def index() -> FileResponse:
    # no-store so the browser always re-fetches index.html (and thus the latest
    # ?v= on the scorer import), preventing stale cached JS after an update.
    return FileResponse(_STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})
