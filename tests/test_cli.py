import json
from types import SimpleNamespace

import pytest

import yva.pipeline
from yva.cli import main


def test_count_passes_args_and_prints(monkeypatch, capsys):
    calls = []

    def fake_count_video(path, p1, p2, model, classes):
        calls.append((path, p1, p2, model, classes))
        return SimpleNamespace(count_in=5, count_out=3)

    # Replace YOLO with a fake, so the test needs no model or real video.
    monkeypatch.setattr(yva.pipeline, "count_video", fake_count_video)
    main(["count", "clip.mp4", "--line", "0", "240", "854", "240", "--classes", "0"])

    assert calls == [("clip.mp4", (0, 240), (854, 240), "yolo11n.pt", [0])]
    assert capsys.readouterr().out.strip() == "in: 5  out: 3"


def test_line_needs_four_numbers():
    with pytest.raises(SystemExit) as e:
        main(["count", "clip.mp4", "--line", "0", "240"])
    assert e.value.code == 2  # argparse's exit code for bad arguments


def test_occupancy_loads_zones_and_prints_shares(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_occupancy_video(path, zones, **kwargs):
        calls.append((path, zones, kwargs))
        return {"A1": 0.5}

    monkeypatch.setattr(yva.pipeline, "occupancy_video", fake_occupancy_video)
    zones_file = tmp_path / "zones.json"
    zones_file.write_text('{"A1": [[0, 0], [10, 0], [10, 10]]}')
    main(["occupancy", "lot.mp4", "--zones", str(zones_file), "--classes", "2", "--max-det", "1000"])

    assert calls == [
        ("lot.mp4", {"A1": [(0, 0), (10, 0), (10, 10)]},
         {"model": "yolo11n.pt", "classes": [2], "imgsz": 640, "max_det": 1000})
    ]
    assert json.loads(capsys.readouterr().out) == {"A1": 0.5}


@pytest.mark.parametrize("content, message", [
    ("{not json", "not valid JSON"),
    ('{"A1": [[0, 0], [10, 0]]}', "at least 3 corners"),
])
def test_bad_zones_file_gives_clear_error(content, message, capsys, tmp_path):
    zones_file = tmp_path / "zones.json"
    zones_file.write_text(content)
    with pytest.raises(SystemExit) as e:
        main(["occupancy", "lot.mp4", "--zones", str(zones_file)])
    assert e.value.code == 2
    assert message in capsys.readouterr().err
