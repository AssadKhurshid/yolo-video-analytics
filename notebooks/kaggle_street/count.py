"""Kaggle job: count vehicles crossing a line in a 30 s street clip (Talaimari/Weast.mp4, 112-142 s).

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_street
Outputs in /kaggle/working: counts.json.
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

from yva.pipeline import count_video

MODEL = "yolo11n.pt"
CLASSES = [1, 2, 3, 5, 7]  # COCO bicycle, car, motorcycle, bus, truck (no rickshaw class in COCO)
START_S, END_S = 112, 142  # steadiest 30 s of the handheld clip
P1, P2 = (0, 480), (1280, 480)  # horizontal line across the road

# Cut the segment into its own file, so the tracker sees exactly the 30 s that was hand-counted.
src = glob.glob("/kaggle/input/**/Talaimari/Weast.mp4", recursive=True)[0]
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.set(cv2.CAP_PROP_POS_FRAMES, round(START_S * fps))
out = cv2.VideoWriter("segment.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
frames = 0
for _ in range(round((END_S - START_S) * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)
    frames += 1
cap.release()
out.release()

start = time.time()
counter = count_video("segment.mp4", P1, P2, model=MODEL, classes=CLASSES)
result = {
    "video": "Talaimari/Weast.mp4",
    "segment_s": [START_S, END_S],
    "frames": frames,
    "fps": fps,
    "line": [P1, P2],
    "model": MODEL,
    "classes": CLASSES,
    "ultralytics": ultralytics.__version__,
    "count_in": counter.count_in,
    "count_out": counter.count_out,
    "seconds": round(time.time() - start, 1),
}
with open("counts.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
