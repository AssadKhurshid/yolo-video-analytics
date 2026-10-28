"""Feed YOLO tracking output into a LineCounter."""

from yva.counting import LineCounter, Point


def update_counter(counter: LineCounter, result) -> None:
    """Pass one frame of ultralytics tracking output to the counter.

    `result` is one ultralytics Results object from model.track(..., persist=True).
    Each box becomes (track ID, box center). Frames where the tracker has no
    IDs yet (boxes.id is None) are skipped.
    """
    boxes = result.boxes
    if boxes.id is None:
        return
    for (x1, y1, x2, y2), track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
        counter.update(track_id, ((x1 + x2) / 2, (y1 + y2) / 2))


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
