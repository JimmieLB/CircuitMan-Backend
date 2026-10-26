"""Move new image/label pairs into the Ultralytics YOLO layout, split randomly.

    dataset/images/{train,val}/*.png
    dataset/labels/{train,val}/*.txt

    python prepare_dataset.py                     # sorts everything in uncategorized/
    python prepare_dataset.py --source some/dir   # or another folder of board_*.png + board_*.txt

Files already in train/val are left where they are.
"""
import argparse
import random
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE / "dataset"
VAL_FRACTION = 0.2


def main(source, seed):
    images = sorted(p for p in source.glob("*.png") if p.with_suffix(".txt").exists())
    orphans = sorted(p.name for p in source.glob("*.png") if not p.with_suffix(".txt").exists())
    if orphans:
        print(f"skipping {len(orphans)} images without labels: {orphans}")
    existing = {p.name for p in (ROOT / "images").glob("*/*.png")}
    dupes = [p.name for p in images if p.name in existing]
    if dupes:
        print(f"skipping {len(dupes)} images already in the dataset: {dupes}")
        images = [p for p in images if p.name not in existing]

    random.Random(seed).shuffle(images)
    n_val = max(1, round(len(images) * VAL_FRACTION)) if images else 0
    splits = {"val": images[:n_val], "train": images[n_val:]}

    for split, files in splits.items():
        (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.move(img, ROOT / "images" / split / img.name)
            shutil.move(img.with_suffix(".txt"), ROOT / "labels" / split / img.with_suffix(".txt").name)
        print(f"{split}: moved {len(files)}, total {len(list((ROOT / 'images' / split).glob('*.png')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=HERE / "uncategorized")
    ap.add_argument("--seed", type=int, default=None, help="set for a reproducible split")
    args = ap.parse_args()
    main(args.source, args.seed)
