# 🧠 training/ — your own kill-detection model

The built-in vision model is generic and not great at spotting kills. This folder is
the **end-to-end pipeline to train your own** and plug it into the app:

```
data preparation → train on Colab → evaluate → export to ONNX → push to Hugging Face → use in the app
```

## What we train (and why)

A small **image classifier** — `kill` vs `no_kill` — *fine-tuned* from a pretrained
vision backbone (default `google/vit-base-patch16-224`).

- **Fine-tuning, not from scratch.** True from-scratch training needs millions of
  labeled images and big GPUs. Fine-tuning reuses a model that already understands
  images and only teaches it your one new distinction — it needs ~1–3k images per
  class and trains in minutes on Colab's free GPU. Same result, far less effort.
- **A classifier, not a chatbot/VLM.** For a yes/no visual question, a classifier is
  more accurate, ~100× smaller, and — importantly — uses the `image-classification`
  task that Transformers.js fully supports in the browser (the VLM path kept
  erroring).

## Contents

| Path | What it is |
|---|---|
| `kill_detector_colab.ipynb` | **The main notebook** — open it in Google Colab and run top to bottom |
| `data/README.md` | How to extract frames from videos and label them |
| `scripts/extract_frames.py` | Video(s) → JPEG frames at a fixed rate |
| `scripts/prepare_dataset.py` | Validate the labeled dataset, check balance, optional split |
| `scripts/predict.py` | Test a trained model locally on an image / folder |
| `deployment/` | How to use the trained model in the app (`scorer_classifier.js` + guide) |
| `requirements.txt` | Deps for the notebook + local scripts |

## Quick start

1. **Open the notebook in Colab:** go to <https://colab.research.google.com>,
   *File → Upload notebook*, pick `kill_detector_colab.ipynb` (or *File → Open
   notebook → GitHub* and paste the repo URL). Set **Runtime → GPU**.
2. **Prove the pipeline first:** leave `DEMO_MODE = True` and run all cells — it
   trains on a tiny public dataset and pushes a model in ~5 minutes, so you know the
   whole train → export → Hugging Face flow works before investing in data.
3. **Build your dataset:** follow `data/README.md` to extract + label frames from your
   gameplay videos.
4. **Train for real:** set `DEMO_MODE = False`, point `DATA_DIR` at your dataset on
   Google Drive, set `HF_REPO_ID` to your Hugging Face username, run all cells.
5. **Use it in the app:** follow `deployment/README.md`.

## Honest expectations

- **Cost:** free. Colab's free T4 GPU and a free Hugging Face account are enough.
- **Time:** training is minutes. **Labeling is the real work** — budget a few hours
  to gather and sort a first dataset. Quality/variety of labels drives quality of
  results far more than model size or epochs.
- **Iterate:** train v1, run it on real clips, find the frames it gets wrong, add
  those to the dataset, retrain. Two or three rounds of this beats any one-shot run.
