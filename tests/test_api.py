from types import SimpleNamespace

from fastapi.testclient import TestClient

import yva.api
from yva.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_count_returns_counts(monkeypatch):
    calls = []

    def fake_count_video(path, p1, p2, classes=None):
        calls.append((p1, p2, classes))
        return SimpleNamespace(count_in=5, count_out=3)

    # Replace YOLO with a fake, so the test needs no model or real video.
    monkeypatch.setattr(yva.api, "count_video", fake_count_video)
    response = client.post(
        "/count",
        files={"video": ("clip.mp4", b"not a real video", "video/mp4")},
        data={"x1": 0, "y1": 240, "x2": 854, "y2": 240, "classes": [0]},
    )
    assert response.json() == {"count_in": 5, "count_out": 3}
    assert calls == [((0, 240), (854, 240), [0])]
