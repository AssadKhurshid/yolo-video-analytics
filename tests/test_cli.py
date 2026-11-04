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
