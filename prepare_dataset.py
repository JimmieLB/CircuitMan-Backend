"""Split loose image/label pairs in dataset/ into the Ultralytics YOLO layout.

    dataset/images/{train,val}/*.png
    dataset/labels/{train,val}/*.txt

Re-run after dropping new board_*.png / board_*.txt pairs into dataset/;
already-split files are left where they are.
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent / "dataset"
VAL_FRACTION = 0.2
SEED = 42


def main():
    images = sorted(p for p in ROOT.glob("*.png") if p.with_suffix(".txt").exists())
    orphans = sorted(p.name for p in ROOT.glob("*.png") if not p.with_suffix(".txt").exists())
    if orphans:
        print(f"skipping {len(orphans)} images without labels: {orphans}")

    random.Random(SEED).shuffle(images)
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
    main()
