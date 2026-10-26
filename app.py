"""HTTP API that runs the breadboard YOLO-pose model.

    uvicorn app:app --reload
    curl -X POST -F "image=@board.png" http://localhost:8000/detect

Set MODEL_PATH to use weights other than runs/pose/breadboard/weights/best.pt.
"""
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent / "runs" / "pose" / "breadboard" / "weights" / "best.pt"))
IMGSZ = 1280      # must match training
# Labels fill a part's real pins into the first keypoint slots, so take the first N for known classes.
PIN_COUNT = {"wire": 2, "resistor": 2, "led": 2, "capacitor": 2, "battery": 2, "transistor": 3, "pot": 3, "button": 4}
PIN_CONF = 0.5    # for classes not listed above, keep slots the model is confident about

if not MODEL_PATH.exists():
    raise RuntimeError(f"model not found at {MODEL_PATH}; set MODEL_PATH to your trained best.pt")
model = YOLO(MODEL_PATH)
if model.task != "pose" or model.model.yaml.get("kpt_shape") != [4, 3]:
    # e.g. the stock COCO yolo11s-pose.pt (17 keypoints, class 'person') rather than our trained weights
    raise RuntimeError(f"{MODEL_PATH} is not a breadboard pin model (classes: {list(model.names.values())})")

app = FastAPI(title="CircuitMax detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "classes": model.names}


@app.post("/detect")
def detect(image: UploadFile = File(...), conf: float = Query(0.4, ge=0, le=1)):
    # Sync handler: FastAPI runs it in a worker thread so inference doesn't block the event loop.
    try:
        image = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "file is not a readable image")

    r = model.predict(image, imgsz=IMGSZ, conf=conf, verbose=False)[0]
    detections = []
    for cls, score, kxy, kconf in zip(
        r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.keypoints.xy.tolist(), r.keypoints.conf.tolist(),
    ):
        name = r.names[int(cls)]
        slots = list(zip(kxy, kconf))
        used = slots[:PIN_COUNT[name]] if name in PIN_COUNT else [s for s in slots if s[1] >= PIN_CONF]
        # Always 4 [x, y] pairs in image pixels, unused slots padded with [0, 0] like the training labels.
        pins = [[round(x), round(y)] for (x, y), _ in used]
        pins += [[0, 0]] * (len(slots) - len(pins))
        detections.append({"type": name, "confidence": round(score, 2), "pins": pins})
    return {"detections": detections}
