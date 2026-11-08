# yolo-video-analytics

What: detection + tracking + line counting + zone occupancy, as a CLI and a FastAPI service.
Replaces three old tutorial repos (car counter, people counter, parking space) with one tool.
Done when: Docker image works, demo GIF in README, count accuracy vs a hand count on 2 clips.

Package: src/yva/   Tests: tests/   Configs: configs/
Commands:
- install: pip install -e ".[dev]"
- test:    pytest
- lint:    ruff check .

Milestones (tick them in the 365 Build Log):
1. Repo skeleton, merge the 3 counter repos
2. Tracking + line counting
3. Zone occupancy mode
4. FastAPI + Docker
5. Count accuracy vs hand count on 2 clips
6. README + demo GIF

Current status: milestone 2 done. First Kaggle run on pedestrian.mp4 (yolo11n, mid line): in 5, out 3, not yet checked by hand. Milestone 3 done: parking (396 spots, drone view) COCO yolo11n = 0 detections; yolo11n-obb @1024 = 256 spots taken >50% of frames with max_det 1000 (250 with the default 300). Milestone 4: FastAPI /health + /count (yva.api) and Dockerfile (CPU torch, model baked in, built + health-checked in CI). Demo GIF in README (milestone 6, part 1). CLI: yva count, yva occupancy (JSON zones). Next: milestone 5, hand count on 2 clips.