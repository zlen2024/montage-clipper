# Building the dataset (the part that actually matters)

The model is only as good as the frames you label. Plan to spend most of your time
here, not on training (training is minutes; labeling is the work).

## Target

A folder with two subfolders — the folder **names are the labels**:

```
dataset/
├── kill/        # the player gets a kill / is in active combat (the stuff you want clipped)
└── no_kill/     # everything else: walking, looting, menus, parachuting, downtime
```

- **Aim for ~1,000–3,000 images per class** to start. More is better.
- Kills are *rarer* than non-kills, so you'll naturally have fewer `kill/` frames —
  that's fine (the notebook uses class-weighted loss), but don't go below ~500.
- **Variety beats volume:** different maps, weapons, skins, times of day, players.
  100 near-identical frames from one fight teaches the model almost nothing.

## Step 1 — Extract frames from your videos

```bash
pip install -r training/requirements.txt          # for the scripts
python training/scripts/extract_frames.py myclips/ --out frames/ --fps 1 --width 512
```

This drops JPEGs into `frames/`, each named like `clip_t42.0.jpg` (source + timestamp).

## Step 2 — Label them (sort into kill/ vs no_kill/)

Make `dataset/kill/` and `dataset/no_kill/`, then drag each frame into the right one.
A file explorer with thumbnails (Windows Explorer, macOS Finder, or an image viewer)
is the fastest manual tool.

**Bootstrapping trick (semi-automatic):** use a model to *pre-sort*, then you only
correct mistakes (this is "active learning" and it's much faster than from-zero):

```bash
# pre-label with any existing classifier or the app's current model, then fix by hand
python training/scripts/predict.py --model some-existing-model --image frames/
```

For PUBG/BGMI specifically, the strongest visual signal of a kill is the **kill-feed
banner** ("knocked out" / "finished") in the corner — when labeling, treat frames
showing that banner (and the surrounding fight) as `kill/`.

## Step 3 — Sanity-check (and optionally split)

```bash
python training/scripts/prepare_dataset.py dataset/
```

It prints class counts, warns about imbalance, and flags tiny datasets. The notebook
makes its own train/val split, so you don't need `--split` unless you want the folders.

## Step 4 — Get it into Colab

Upload the `dataset/` folder to your **Google Drive** (e.g.
`MyDrive/montage_dataset`), then in the notebook set `DEMO_MODE = False` and
`DATA_DIR = "/content/drive/MyDrive/montage_dataset"`.

> ⚖️ **Only use footage you have the right to use** — your own recordings, or clips
> you have permission for. Respect game and platform terms.
