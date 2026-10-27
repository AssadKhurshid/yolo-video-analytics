"""Count tracked objects crossing a line."""

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    """2D cross product of (a - o) and (b - o). Its sign says which side of o->a b is on."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class LineCounter:
    """Counts tracks whose centroid crosses the segment p1 -> p2.

    The side of a point p is the sign of cross(p2 - p1, p - p1). In image
    coordinates (y points down) the positive side is on your right when you
    walk from p1 to p2. Example: for p1=(0, 0), p2=(10, 0) it is below the line.
      "in"  = crossing from the negative side to the positive side
      "out" = crossing from the positive side to the negative side

    Each track is counted at most once per direction: a track that goes in,
    out and in again gives count_in=1, count_out=1.

    A centroid exactly on the line (cross == 0) is skipped. The crossing is
    counted when the track reaches the other side.
    """

    def __init__(self, p1: Point, p2: Point):
        if p1 == p2:
            raise ValueError(f"line needs two different points, got p1 == p2 == {p1}")
        self.p1, self.p2 = p1, p2
        self._last: dict[int, Point] = {}  # last centroid that was off the line, per track
        self._in: set[int] = set()
        self._out: set[int] = set()

    @property
    def count_in(self) -> int:
        return len(self._in)

    @property
    def count_out(self) -> int:
        return len(self._out)

    def update(self, track_id: int, centroid: Point) -> None:
        side = _cross(self.p1, self.p2, centroid)
        if side == 0:
            return
        prev = self._last.get(track_id)
        self._last[track_id] = centroid
        if prev is None or (_cross(self.p1, self.p2, prev) > 0) == (side > 0):
            return
        # Sides changed. The step prev -> centroid must also pass between p1 and p2,
        # not beyond the ends of the segment: p1 and p2 must not be on the same side of it.
        if _cross(prev, centroid, self.p1) * _cross(prev, centroid, self.p2) > 0:
            return
        (self._in if side > 0 else self._out).add(track_id)
