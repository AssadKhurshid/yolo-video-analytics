"""Command line:
yva count VIDEO --line X1 Y1 X2 Y2 [--classes 0] [--model yolo11n.pt]
yva occupancy VIDEO --zones zones.json [--classes 2] [--model yolo11n.pt] [--imgsz 640] [--max-det 300]
"""

import argparse
import json

from yva import pipeline
from yva.counting import Point


def load_zones(path: str) -> dict[str, list[Point]]:
    """Read {"zone name": [[x, y], ...]} from a JSON file. Raises ValueError if it's unusable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected zone names mapped to corner lists")

    zones = {}
    for name, corners in data.items():
        try:
            zones[name] = [(float(x), float(y)) for x, y in corners]
        except (TypeError, ValueError) as e:
            raise ValueError(f"zone {name}: corners must be [x, y] pairs") from e
        if len(zones[name]) < 3:
            raise ValueError(f"zone {name}: needs at least 3 corners, got {len(zones[name])}")
    return zones


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yva", description="Video analytics with YOLO + tracking.")
    commands = parser.add_subparsers(dest="command", required=True)

    count = commands.add_parser("count", help="count objects crossing a line")
    count.add_argument("video", help="path to a video file")
    count.add_argument("--line", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"),
                       help="line end points in pixels")
    count.add_argument("--classes", nargs="+", type=int, help="COCO class IDs to keep, e.g. 0 for people")
    count.add_argument("--model", default="yolo11n.pt", help="YOLO model file (default: %(default)s)")

    occupancy = commands.add_parser("occupancy", help="share of frames each zone is occupied")
    occupancy.add_argument("video", help="path to a video file")
    occupancy.add_argument("--zones", required=True, help='JSON file: {"name": [[x, y], ...]} in pixels')
    occupancy.add_argument("--classes", nargs="+", type=int, help="class IDs to keep, e.g. 2 for COCO cars")
    occupancy.add_argument("--model", default="yolo11n.pt", help="YOLO model file (default: %(default)s)")
    occupancy.add_argument("--imgsz", type=int, default=640, help="YOLO input size (default: %(default)s)")
    occupancy.add_argument("--max-det", type=int, default=300, help="max detections per frame (default: %(default)s)")

    args = parser.parse_args(argv)
    if args.command == "count":
        x1, y1, x2, y2 = args.line
        counter = pipeline.count_video(args.video, (x1, y1), (x2, y2), model=args.model, classes=args.classes)
        print(f"in: {counter.count_in}  out: {counter.count_out}")
    else:
        try:
            zones = load_zones(args.zones)
        except ValueError as e:
            parser.error(str(e))  # prints the message and exits with code 2
        shares = pipeline.occupancy_video(
            args.video, zones, model=args.model, classes=args.classes, imgsz=args.imgsz, max_det=args.max_det
        )
        print(json.dumps(shares, indent=2))
