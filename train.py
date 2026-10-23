"""Train a YOLO-pose model to find breadboard components and their pins.

    pip install ultralytics
    python train.py                      # train (Ctrl+C to pause; last.pt is saved every epoch)
    python train.py --resume             # continue a paused run from its last finished epoch
    python train.py --predict some.png   # run the best checkpoint on an image
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DATA = (Path(__file__).parent / "dataset" / "data.yaml").resolve()
RUNS = (Path(__file__).parent / "runs" / "pose").resolve()  # absolute, or Ultralytics nests it under its own runs/pose
BEST = RUNS / "breadboard" / "weights" / "best.pt"
LAST = BEST.with_name("last.pt")


def train(model_name, epochs, device):
    model = YOLO(model_name)  # COCO-pretrained weights; the head is re-initialized for our classes/keypoints
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=1280,      # images are 1300x420 and parts are ~40-80 px wide; don't downscale them away
        batch=4,
        patience=50,
        # No flips: 2-pin parts are labelled left-end-first, and a flip would reverse that order
        # (one global flip_idx can't fix it without also scrambling transistor pins).
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        close_mosaic=20,
        project=str(RUNS),
        name="breadboard",
        exist_ok=True,
        device=device,
    )
    model.val(data=str(DATA))


def predict(image):
    model = YOLO(BEST)
    for r in model.predict(image, imgsz=1280, conf=0.4, save=True):
        for box, cls, kpts in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.keypoints.xy.tolist()):
            print(r.names[int(cls)], [round(v) for v in box], "pins:", [[round(x), round(y)] for x, y in kpts if x or y])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-pose.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default=None, help="0 for first GPU, 'cpu', or 'mps'")
    ap.add_argument("--predict", help="image to run inference on instead of training")
    ap.add_argument("--resume", action="store_true", help="continue the interrupted run from last.pt")
    args = ap.parse_args()
    if args.predict:
        predict(args.predict)
    elif args.resume:
        YOLO(LAST).train(resume=True)  # restores the run's original settings, epoch and optimizer state
    else:
        train(args.model, args.epochs, args.device)
