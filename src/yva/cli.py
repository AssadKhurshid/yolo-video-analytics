"""Command line: yva count VIDEO --line X1 Y1 X2 Y2 [--classes 0] [--model yolo11n.pt]"""

import argparse

from yva import pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yva", description="Video analytics with YOLO + tracking.")
    commands = parser.add_subparsers(dest="command", required=True)

    count = commands.add_parser("count", help="count objects crossing a line")
    count.add_argument("video", help="path to a video file")
    count.add_argument("--line", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"),
                       help="line end points in pixels")
    count.add_argument("--classes", nargs="+", type=int, help="COCO class IDs to keep, e.g. 0 for people")
    count.add_argument("--model", default="yolo11n.pt", help="YOLO model file (default: %(default)s)")

    args = parser.parse_args(argv)
    x1, y1, x2, y2 = args.line
    counter = pipeline.count_video(args.video, (x1, y1), (x2, y2), model=args.model, classes=args.classes)
    print(f"in: {counter.count_in}  out: {counter.count_out}")
