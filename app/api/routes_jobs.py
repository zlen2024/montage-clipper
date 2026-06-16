"""Job endpoints: upload, status, frame manifest/images, score submission, download."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.api.schemas import (
    FramesManifest,
    FrameSegment,
    JobStatusResponse,
    ScoreSubmission,
    UploadResponse,
)
from app.config import Settings, get_settings
from app.core.credits import CreditError, CreditService
from app.core.jobs import Job, JobStatus, JobStore, Tier
from app.core.pipeline import run as run_pipeline
from app.deps import get_credits, get_store
from app.media.probe import probe_duration

router = APIRouter()

_ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_CHUNK = 1024 * 1024


def _account_id(x_account_id: str | None) -> str:
    return x_account_id or "anonymous"


async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(_CHUNK):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Upload exceeds size limit.")
            f.write(chunk)


@router.post("/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    tier: str = Form("free"),
    x_account_id: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    store: JobStore = Depends(get_store),
    credits: CreditService = Depends(get_credits),
) -> UploadResponse:
    try:
        tier_enum = Tier(tier)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown tier '{tier}'.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXT)}",
        )

    # Gate by tier/credits before doing any work.
    try:
        credits.authorize(_account_id(x_account_id), tier_enum)
    except CreditError as exc:
        raise HTTPException(status_code=402, detail=str(exc))

    job_id = uuid.uuid4().hex
    source = settings.uploads_dir / job_id / f"source{ext}"
    await _save_upload(file, source, settings.max_upload_bytes)

    duration = probe_duration(source)
    if duration > settings.max_duration_seconds:
        source.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Video is {duration:.0f}s; max is {settings.max_duration_seconds:.0f}s.",
        )

    job = store.create(Job(id=job_id, tier=tier_enum, filename=file.filename or f"source{ext}"))
    background_tasks.add_task(run_pipeline, job_id, source, store, settings)
    return UploadResponse(job_id=job_id, status=job.status.value, tier=tier_enum.value)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, store: JobStore = Depends(get_store)) -> JobStatusResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(**job.public_dict())


@router.get("/jobs/{job_id}/frames", response_model=FramesManifest)
def frames_manifest(
    job_id: str,
    store: JobStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> FramesManifest:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != JobStatus.AWAITING_CLIENT_SCORING:
        raise HTTPException(status_code=409, detail="Job is not awaiting client scoring.")

    segments = [
        FrameSegment(
            index=i,
            start=seg.start,
            end=seg.end,
            frame_urls=[
                f"/jobs/{job_id}/frames/{i}/{n}.jpg"
                for n in range(1, job.frame_counts[i] + 1)
            ],
        )
        for i, seg in enumerate(job.segments)
    ]
    return FramesManifest(job_id=job_id, model_id=settings.vlm_model_id, segments=segments)


@router.get("/jobs/{job_id}/frames/{seg}/{n}.jpg")
def frame_image(
    job_id: str,
    seg: int,
    n: int,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    path = settings.work_dir / job_id / f"seg_{seg:03d}" / f"frame_{n:03d}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/jobs/{job_id}/scores")
def submit_scores(
    job_id: str,
    submission: ScoreSubmission,
    store: JobStore = Depends(get_store),
) -> dict[str, str]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != JobStatus.AWAITING_CLIENT_SCORING:
        raise HTTPException(status_code=409, detail="Job is not awaiting client scoring.")

    try:
        scores = {int(k): float(v) for k, v in submission.scores.items()}
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Scores must map integer indices to numbers.")

    store.submit_scores(job_id, scores)
    return {"status": "accepted"}


@router.get("/jobs/{job_id}/download")
def download(job_id: str, settings: Settings = Depends(get_settings),
             store: JobStore = Depends(get_store)) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job is '{job.status.value}', not ready.")

    path = settings.outputs_dir / job_id / "montage.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Montage file missing.")
    return FileResponse(path, media_type="video/mp4", filename="montage.mp4")
