"""Kaggle job: count people crossing a line in pedestrian.mp4.

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_count
Outputs in /kaggle/working: counts.json, line.jpg (first frame with the line drawn).
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
video = glob.glob("/kaggle/input/**/pedestrian.mp4", recursive=True)[0]

cap = cv2.VideoCapture(video)
ok, first_frame = cap.read()
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, n_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# First run: horizontal line across the middle. Move it once we've seen line.jpg.
p1, p2 = (0, h // 2), (w, h // 2)

start = time.time()
counter = count_video(video, p1, p2, model=MODEL, classes=[0])  # COCO class 0 = person
seconds = time.time() - start

result = {
    "video": "pedestrian.mp4",
    "size": [w, h],
    "fps": fps,
    "frames": n_frames,
    "line": [p1, p2],
    "model": MODEL,
    "ultralytics": ultralytics.__version__,
    "count_in": counter.count_in,
    "count_out": counter.count_out,
    "seconds": round(seconds, 1),
}
with open("counts.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))

cv2.line(first_frame, p1, p2, (0, 0, 255), 3)
cv2.imwrite("line.jpg", first_frame)
