"""Feed YOLO output into a LineCounter or zone occupancy."""

from collections.abc import Iterable

from yva.counting import LineCounter, Point
from yva.zones import occupancy


def box_centers(result) -> list[Point]:
    """Center (x, y) of every box in one frame of ultralytics output."""
    return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in result.boxes.xyxy.tolist()]


def update_counter(counter: LineCounter, result) -> None:
    """Pass one frame of ultralytics tracking output to the counter.

    `result` is one ultralytics Results object from model.track(..., persist=True).
    Each box becomes (track ID, box center). Frames where the tracker has no
    IDs yet (boxes.id is None) are skipped.
    """
    if result.boxes.id is None:
        return
    for track_id, center in zip(result.boxes.id.int().tolist(), box_centers(result)):
        counter.update(track_id, center)


def zone_shares(zones: dict[str, list[Point]], results: Iterable) -> dict[str, float]:
    """Share of frames (0 to 1) in which each zone holds at least one box center."""
    occupied = dict.fromkeys(zones, 0)
    frames = 0
    for result in results:
        frames += 1
        for name, taken in occupancy(zones, box_centers(result)).items():
            occupied[name] += taken
    return {name: n / frames for name, n in occupied.items()}


def count_video(
    source: str, p1: Point, p2: Point, model: str = "yolo11n.pt", classes: list[int] | None = None
) -> LineCounter:
    """Detect + track objects in a video and count line crossings.

    `classes` are COCO class IDs to keep, e.g. [0] people or [2, 3, 5, 7] vehicles.
    Needs the video extra: pip install -e ".[video]".
    """
    from ultralytics import YOLO  # imported here so tests and CI don't need ultralytics

    counter = LineCounter(p1, p2)
    for result in YOLO(model).track(source, stream=True, persist=True, classes=classes, verbose=False):
        update_counter(counter, result)
    return counter


def occupancy_video(
    source: str,
    zones: dict[str, list[Point]],
    model: str = "yolo11n.pt",
    classes: list[int] | None = None,
    imgsz: int = 640,
) -> dict[str, float]:
    """Detect objects in a video and return each zone's share of frames occupied.

    No tracker: "is the zone taken?" only needs detections in each frame.
    `imgsz` is the size YOLO resizes frames to; raise it for small objects
    (e.g. cars seen from a drone). Needs the video extra: pip install -e ".[video]".
    """
    from ultralytics import YOLO

    results = YOLO(model).predict(source, stream=True, classes=classes, imgsz=imgsz, verbose=False)
    return zone_shares(zones, results)
