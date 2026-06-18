#!/usr/bin/env python3
"""Quick local inference test for a trained kill detector.

Point it at a local model folder or a Hugging Face repo id, and an image or a
folder of images.

Usage
-----
    python predict.py --model your-username/montage-kill-detector --image frame.jpg
    python predict.py --model ./kill-detector --image frames/ --top 1

Needs: pip install -r training/requirements.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Local model dir or HF repo id")
    ap.add_argument("--image", type=Path, required=True, help="Image file or folder")
    ap.add_argument("--top", type=int, default=2, help="Top-k labels to print")
    args = ap.parse_args()

    try:
        from transformers import pipeline
    except ImportError:
        sys.exit("transformers not installed. Run: pip install -r training/requirements.txt")

    clf = pipeline("image-classification", model=args.model)

    if args.image.is_dir():
        images = [p for p in sorted(args.image.iterdir()) if p.suffix.lower() in IMG_EXTS]
    else:
        images = [args.image]

    for img in images:
        preds = clf(str(img), top_k=args.top)
        summary = ", ".join(f"{p['label']}={p['score']:.2f}" for p in preds)
        print(f"{img.name}: {summary}")


if __name__ == "__main__":
    main()
