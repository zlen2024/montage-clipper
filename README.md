# 🎮 Montage Clipper

Upload a long gameplay recording (up to ~1 hour) and get back a short highlight
montage — the app trims it down to the epic moments automatically, so you don't have
to scrub the whole thing to make content.

## How it works

```
upload → extract audio → audio-loudness detect → (optional AI re-rank) → trim+concat → download
```

Two detection tiers (also the monetization seam):

- **Free — audio-loudness detection.** The action (gunfire, explosions, the player
  reacting) is almost always the loudest part. We compute a smoothed energy envelope,
  pick prominent peaks, pad/merge them into segments, cap the total length, and stitch
  with ffmpeg. Game-agnostic, CPU-only, free to run.
- **Paid/credit — AI scene scoring, in your browser.** Frames around the audio
  candidates are scored by an open-weights vision-language model
  (`SmolVLM-500M-Instruct`) running **client-side via WebGPU** — on the user's GPU, so
  there's **no cloud inference cost**. The server only coordinates and renders. The
  `SceneScorer` seam leaves a cloud path (e.g. Claude vision) as a future drop-in.

## Requirements

- Python 3.11+
- **ffmpeg / ffprobe** on PATH (the Docker image installs them; locally:
  `apt-get install ffmpeg` / `brew install ffmpeg`)
- For the AI tier: a WebGPU-capable browser (recent Chrome/Edge)

## Run locally

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
# open http://localhost:8000  (API docs at /docs)
```

## Run with Docker

```bash
docker compose up --build
# open http://localhost:8000
```

## Test

```bash
pytest          # video tests auto-skip if ffmpeg isn't installed
```

The suite synthesizes its own clip with loud bursts at known timestamps, so detection
can be asserted exactly without a real gameplay video (see `samples/README.md`).

## Configuration

All tunables live in `app/config.py` and can be overridden via environment variables
or a `.env` file (see `.env.example`): montage length cap, audio-detection
parameters, the VLM model id, free-tier daily limit, retention, and upload limits.

## Project layout

| Path | Role |
|---|---|
| `app/detect/audio_loudness.py` | free-tier algorithm (the heart of the MVP) |
| `app/media/ffmpeg_io.py` | all extract/sample/trim/concat ffmpeg calls |
| `app/core/pipeline.py` | orchestrates detect → refine → render, drives job status |
| `app/llm/interface.py` | `SceneScorer` seam (client / stub / future cloud) |
| `app/static/scorer.js` | Transformers.js + WebGPU SmolVLM scoring in the browser |
| `app/api/routes_jobs.py` | upload / status / frames / scores / download endpoints |

## Status

MVP. Job processing uses FastAPI background tasks + an in-process store with a JSON
sidecar; the store and queue are kept behind small interfaces so they can move to
Redis + RQ workers without touching the pipeline. Revenue (ads + VIP/credits) and
real auth/payments are future work — the credit gating is stubbed with clean seams.
