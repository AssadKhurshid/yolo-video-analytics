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

Current status: skeleton created. Next: milestone 1.