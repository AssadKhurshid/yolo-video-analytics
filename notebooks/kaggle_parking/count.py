"""Kaggle job: parking spot occupancy on parking_1920_1080.mp4 (drone view, 396 spots).

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_parking
Outputs in /kaggle/working: occupancy.json, first_frame.jpg (red = taken, green = free).
"""

import glob
import json
import subprocess
import sys
import time

# Installs the latest main from GitHub, so push code before starting a run.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "yolo-video-analytics[video] @ git+https://github.com/AssadKhurshid/yolo-video-analytics.git",
    ],
    check=True,
)

import cv2  # installed with ultralytics above
import ultralytics
from ultralytics import YOLO

from yva.pipeline import box_centers, occupancy_video
from yva.zones import occupancy

MODEL = "yolo11n.pt"
CLASSES = [2, 5, 7]  # COCO car, bus, truck
video = glob.glob("/kaggle/input/**/parking_1920_1080.mp4", recursive=True)[0]
mask = cv2.imread(glob.glob("/kaggle/input/**/mask_1920_1080.png", recursive=True)[0], cv2.IMREAD_GRAYSCALE)

# One zone per white blob in the mask: its bounding box as a 4-corner polygon.
_, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype("uint8"))
zones = {
    f"S{i:03d}": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i, (x, y, w, h, _) in enumerate(stats[1:].tolist(), start=1)
}

runs = []
for imgsz in (640, 1920):  # YOLO's default input size vs full frame width
    start = time.time()
    shares = occupancy_video(video, zones, model=MODEL, classes=CLASSES, imgsz=imgsz)
    runs.append({
        "imgsz": imgsz,
        "mean_share_occupied": round(sum(shares.values()) / len(shares), 3),
        "spots_occupied_over_half_the_time": sum(s > 0.5 for s in shares.values()),
        "seconds": round(time.time() - start, 1),
    })

# First frame at full size: which spots are taken, drawn for the README.
cap = cv2.VideoCapture(video)
_, frame = cap.read()
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
first = YOLO(MODEL).predict(frame, classes=CLASSES, imgsz=1920, verbose=False)[0]
taken = occupancy(zones, box_centers(first))
for name, poly in zones.items():
    cv2.rectangle(frame, poly[0], poly[2], (0, 0, 255) if taken[name] else (0, 200, 0), 2)
cv2.imwrite("first_frame.jpg", frame)

result = {
    "video": "parking_1920_1080.mp4",
    "frames": n_frames,
    "spots": len(zones),
    "model": MODEL,
    "ultralytics": ultralytics.__version__,
    "runs": runs,
    "first_frame_imgsz_1920": {"detections": len(first.boxes), "spots_taken": sum(taken.values())},
}
with open("occupancy.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
