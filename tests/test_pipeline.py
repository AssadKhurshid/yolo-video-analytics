import sys
from types import SimpleNamespace

from yva.counting import LineCounter
from yva.pipeline import box_centers, occupancy_video, update_counter, zone_shares


class FakeTensor:
    """Just enough of a torch tensor for update_counter: .tolist() and .int()."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data

    def int(self):
        return FakeTensor([int(v) for v in self.data])


def frame(boxes, ids):
    """Fake ultralytics Results with xyxy boxes and float track IDs (or None)."""
    id_tensor = None if ids is None else FakeTensor(ids)
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=FakeTensor(boxes), id=id_tensor))


def test_box_centers_reach_counter():
    c = LineCounter((0, 0), (100, 0))
    # Track 7: box centered at (50, -10), then at (50, 10) -> crosses the line downward.
    update_counter(c, frame([[40, -20, 60, 0]], [7.0]))
    update_counter(c, frame([[40, 0, 60, 20]], [7.0]))
    assert (c.count_in, c.count_out) == (1, 0)


def test_frame_without_track_ids_is_skipped():
    c = LineCounter((0, 0), (100, 0))
    update_counter(c, frame([[40, -20, 60, 0]], None))
    update_counter(c, frame([[40, 0, 60, 20]], [7.0]))
    assert (c.count_in, c.count_out) == (0, 0)  # first sighting of track 7 is below the line


ZONES = {"A1": [(0, 0), (10, 0), (10, 10), (0, 10)], "A2": [(20, 0), (30, 0), (30, 10), (20, 10)]}


def test_zone_taken_in_one_of_two_frames():
    frames = [frame([[2, 2, 8, 8]], None), frame([[22, 2, 28, 8]], None)]  # car in A1, then in A2
    assert zone_shares(ZONES, frames) == {"A1": 0.5, "A2": 0.5}


def test_box_centers_reads_rotated_boxes():
    # OBB models leave boxes as None and put the boxes' axis-aligned bounds in obb.xyxy.
    obb_frame = SimpleNamespace(boxes=None, obb=SimpleNamespace(xyxy=FakeTensor([[0, 0, 10, 4]])))
    assert box_centers(obb_frame) == [(5, 2)]


def test_occupancy_video_passes_max_det_to_yolo(monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, model):
            pass

        def predict(self, source, **kwargs):
            calls.append(kwargs)
            return [frame([], None)]

    # occupancy_video imports ultralytics when called, so a fake module replaces it here.
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    occupancy_video("clip.mp4", ZONES, max_det=1000)
    assert calls[0]["max_det"] == 1000


def test_frame_without_detections_leaves_zones_free():
    assert zone_shares(ZONES, [frame([], None)]) == {"A1": 0.0, "A2": 0.0}
