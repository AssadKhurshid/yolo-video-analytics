from yva.zones import occupancy, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
# L shape: the square with its top-right quarter cut out.
L_SHAPE = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]


def test_inside_and_outside_square():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)


def test_concave_polygon():
    assert point_in_polygon((2, 8), L_SHAPE)
    assert not point_in_polygon((8, 8), L_SHAPE)  # in the cut-out corner


def test_two_cars_in_one_zone_is_one_occupied_zone():
    zones = {"A1": SQUARE, "A2": [(20, 0), (30, 0), (30, 10), (20, 10)]}
    result = occupancy(zones, [(3, 3), (7, 7)])
    assert result == {"A1": True, "A2": False}
    assert sum(not v for v in result.values()) == 1  # free zones


def test_empty_frame_all_free():
    assert occupancy({"A1": SQUARE}, []) == {"A1": False}
