#!/usr/bin/env python3
"""Validate a labeled dataset and (optionally) make a train/val split.

Expects a folder with one subfolder per class, e.g.::

    dataset/
    ├── kill/
    └── no_kill/

Usage
-----
    # just report counts / balance / sanity
    python prepare_dataset.py dataset/

    # also write a stratified split to dataset_split/{train,val}/<class>/
    python prepare_dataset.py dataset/ --split --out dataset_split --val 0.15
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def class_dirs(root: Path) -> list[Path]:
    return [d for d in sorted(root.iterdir()) if d.is_dir()]


def images_in(d: Path) -> list[Path]:
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def report(root: Path) -> dict[str, list[Path]]:
    classes = class_dirs(root)
    if len(classes) < 2:
        sys.exit(f"Need at least 2 class subfolders in {root} (e.g. kill/ and no_kill/).")

    by_class = {c.name: images_in(c) for c in classes}
    print("Class counts:")
    for name, imgs in by_class.items():
        print(f"  {name:12s} {len(imgs)}")

    counts = [len(v) for v in by_class.values()]
    if min(counts) == 0:
        sys.exit("A class has 0 images — add some before training.")
    if min(counts) < 100:
        print("\n⚠ Fewer than 100 images in a class. Aim for ~1,000–3,000 per class for good results.")
    imbalance = max(counts) / min(counts)
    if imbalance > 3:
        print(f"\n⚠ Class imbalance {imbalance:.1f}:1 — the notebook uses weighted loss, "
              "but try to collect more of the rare class.")
    return by_class


def make_split(by_class: dict[str, list[Path]], out: Path, val_frac: float, seed: int) -> None:
    random.seed(seed)
    for split in ("train", "val"):
        for name in by_class:
            (out / split / name).mkdir(parents=True, exist_ok=True)

    for name, imgs in by_class.items():
        imgs = imgs[:]
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_frac))
        for i, src in enumerate(imgs):
            split = "val" if i < n_val else "train"
            shutil.copy(src, out / split / name / src.name)
    print(f"\nWrote split to {out}/ (train + val)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Dataset root (folder of class subfolders)")
    ap.add_argument("--split", action="store_true", help="Also write a train/val split")
    ap.add_argument("--out", type=Path, default=Path("dataset_split"), help="Output dir for --split")
    ap.add_argument("--val", type=float, default=0.15, help="Validation fraction (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_class = report(args.root)
    if args.split:
        make_split(by_class, args.out, args.val, args.seed)


if __name__ == "__main__":
    main()
