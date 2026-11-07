"""Kaggle job: demo GIF of the line counter on pedestrian.mp4.

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_demo
Outputs in /kaggle/working: demo.gif, demo.json (final counts of this run).
"""

import glob
import json
import os
import subprocess
import sys

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
from PIL import Image
from ultralytics import YOLO

from yva.counting import LineCounter
from yva.pipeline import update_counter

MODEL = "yolo11n.pt"
STEP = 6  # keep every 6th frame in the GIF; the counter still sees every frame
SIZE = (400, 225)  # GIF picture size; the clip is 854x480
FRAME_MS = 125  # 8 fps: with STEP = 6 on a 24 fps clip, plays at 2x speed
RED, GREEN, WHITE, BLACK = (0, 0, 255), (0, 200, 0), (255, 255, 255), (0, 0, 0)

video = glob.glob("/kaggle/input/**/pedestrian.mp4", recursive=True)[0]
p1, p2 = (0, 240), (854, 240)  # same line as the results table
counter = LineCounter(p1, p2)

frames = []
tracked = YOLO(MODEL).track(video, stream=True, persist=True, classes=[0], verbose=False)
for i, result in enumerate(tracked):
    update_counter(counter, result)
    if i % STEP:
        continue
    img = result.orig_img.copy()
    cv2.line(img, p1, p2, RED, 3)
    if result.boxes.id is not None:
        for (x1, y1, x2, y2), track_id in zip(result.boxes.xyxy.int().tolist(), result.boxes.id.int().tolist()):
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3)
            cv2.putText(img, str(track_id), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
    text = f"in: {counter.count_in}  out: {counter.count_out}"
    cv2.rectangle(img, (0, 0), (380, 60), BLACK, -1)
    cv2.putText(img, text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 3)
    small = cv2.cvtColor(cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(small).quantize(32))

frames[0].save("demo.gif", save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)

result = {
    "video": "pedestrian.mp4",
    "line": [p1, p2],
    "model": MODEL,
    "ultralytics": ultralytics.__version__,
    "count_in": counter.count_in,
    "count_out": counter.count_out,
    "gif_frames": len(frames),
    "gif_mb": round(os.path.getsize("demo.gif") / 1e6, 2),
}
with open("demo.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
