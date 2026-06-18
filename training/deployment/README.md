# Using your trained model in the app

After the notebook pushes your model to Hugging Face, wire it into Montage Clipper.

## 1. Swap the browser scorer

The app currently uses a generic vision-language model (`app/static/scorer.js`).
Replace it with the classifier scorer, which loads *your* model via the reliable
`image-classification` pipeline:

```bash
cp training/deployment/scorer_classifier.js app/static/scorer.js
```

Commit + push so CI rebuilds the Docker image (or rebuild locally).

## 2. Point the app at your model

The model id is read from the `VLM_MODEL_ID` setting, which the server hands to the
browser in the frames manifest. Set it to your repo:

```bash
docker run --rm -p 8080:8000 \
  -e BYPASS_CREDITS=true \
  -e VLM_MODEL_ID="your-username/montage-kill-detector" \
  ghcr.io/zlen2024/montage-clipper:latest
```

Open <http://localhost:8080>, choose **AI mode**, upload a clip. Each sampled frame is
now classified by your kill detector; frames it scores ≥ `AI_KILL_THRESHOLD` (default
0.5) become montage clips.

## 3. Tune

| Env var | Effect |
|---|---|
| `AI_SAMPLE_PERIOD_S=1.0` | Check every 1 s instead of 2 s (finer, slower) |
| `AI_KILL_THRESHOLD=0.6` | Be stricter about what counts as a kill |
| `MAX_MONTAGE_LEN=120` | Allow a longer montage |

## How it fits together

```
server: sample 1 frame / N seconds  ──►  browser: YOUR classifier scores each frame
   ▲                                                        │
   └──────────  clip every frame scored ≥ threshold ◄───────┘
                merge overlaps → cap length → render montage
```

The model requirements: a label named **`kill`** (the notebook's `kill` / `no_kill`
classes satisfy this) and an `onnx/` folder in the repo (the notebook creates it).

## Why a classifier instead of the VLM?

- **Reliable:** `image-classification` is fully supported in Transformers.js; the
  `image-text-to-text` VLM path is not, which is what kept erroring.
- **Accurate:** a model fine-tuned on *your* game beats a generic model at "is this a
  kill?".
- **Fast & small:** a quantized small classifier is a fraction of a VLM's size and
  runs in a few ms per frame.
