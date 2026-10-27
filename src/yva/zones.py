"""Zone occupancy: which polygons contain at least one object centroid."""

from yva.counting import Point


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    """Ray casting: cast a ray from pt to the right and count the edges it crosses.

    An odd count means pt is inside. Works for concave polygons. Points exactly
    on an edge may land either way.
    """
    x, y = pt
    inside = False
    for (xi, yi), (xj, yj) in zip(poly, poly[1:] + poly[:1]):
        # Edge straddles the ray's y, and the crossing point is right of pt.
        if (yi > y) != (yj > y) and x < xi + (y - yi) * (xj - xi) / (yj - yi):
            inside = not inside
    return inside


def occupancy(zones: dict[str, list[Point]], centroids: list[Point]) -> dict[str, bool]:
    """Map each zone name to True if at least one centroid is inside it.

    Two objects in one zone still make one occupied zone.
    Free zones: sum(not v for v in result.values()).
    """
    return {name: any(point_in_polygon(c, poly) for c in centroids) for name, poly in zones.items()}
