"""Kaggle job: parking spot occupancy on parking_1920_1080.mp4 (drone view, 396 spots).

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_parking
Outputs in /kaggle/working: occupancy.json, first_frame.jpg (aerial model; red = taken, green = free).
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

# (model, classes, input size, max detections per frame).
# COCO: 2 car, 5 bus, 7 truck. DOTA (aerial): 9 large, 10 small vehicle.
RUNS = [
    ("yolo11n.pt", [2, 5, 7], 640, 300),
    ("yolo11n.pt", [2, 5, 7], 1920, 300),
    ("yolo11n-obb.pt", [9, 10], 1024, 300),  # OBB models are trained at 1024
    ("yolo11n-obb.pt", [9, 10], 1920, 300),
    ("yolo11n-obb.pt", [9, 10], 1024, 1000),  # 300 is ultralytics' default cap
]
video = glob.glob("/kaggle/input/**/parking_1920_1080.mp4", recursive=True)[0]
mask = cv2.imread(glob.glob("/kaggle/input/**/mask_1920_1080.png", recursive=True)[0], cv2.IMREAD_GRAYSCALE)

# One zone per white blob in the mask: its bounding box as a 4-corner polygon.
_, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype("uint8"))
zones = {
    f"S{i:03d}": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i, (x, y, w, h, _) in enumerate(stats[1:].tolist(), start=1)
}

runs = []
for model, classes, imgsz, max_det in RUNS:
    start = time.time()
    shares = occupancy_video(video, zones, model=model, classes=classes, imgsz=imgsz, max_det=max_det)
    runs.append({
        "model": model,
        "imgsz": imgsz,
        "max_det": max_det,
        "mean_share_occupied": round(sum(shares.values()) / len(shares), 3),
        "spots_occupied_over_half_the_time": sum(s > 0.5 for s in shares.values()),
        "seconds": round(time.time() - start, 1),
    })

# First frame with the aerial model: which spots are taken, drawn for the README.
cap = cv2.VideoCapture(video)
_, frame = cap.read()
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
first = YOLO("yolo11n-obb.pt").predict(frame, classes=[9, 10], imgsz=1024, max_det=1000, verbose=False)[0]
taken = occupancy(zones, box_centers(first))
for name, poly in zones.items():
    cv2.rectangle(frame, poly[0], poly[2], (0, 0, 255) if taken[name] else (0, 200, 0), 2)
cv2.imwrite("first_frame.jpg", frame)

result = {
    "video": "parking_1920_1080.mp4",
    "frames": n_frames,
    "spots": len(zones),
    "ultralytics": ultralytics.__version__,
    "runs": runs,
    "first_frame_obb_1024_maxdet_1000": {"detections": len(box_centers(first)), "spots_taken": sum(taken.values())},
}
with open("occupancy.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
