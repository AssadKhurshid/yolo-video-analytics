import pytest

from yva.counting import LineCounter


def make_counter():
    # Horizontal line from (0, 0) to (10, 0); in image coords "in" = moving down.
    return LineCounter((0, 0), (10, 0))


def test_crossing_is_counted():
    c = make_counter()
    c.update(1, (5, -5))
    c.update(1, (5, 5))
    assert (c.count_in, c.count_out) == (1, 0)


def test_direction():
    c = make_counter()
    c.update(1, (5, 5))
    c.update(1, (5, -5))
    assert (c.count_in, c.count_out) == (0, 1)


def test_no_double_count():
    c = make_counter()
    for y in (-5, 5, -5, 5, -5):  # in, out, in, out
        c.update(1, (5, y))
    assert (c.count_in, c.count_out) == (1, 1)


def test_miss_outside_segment():
    c = make_counter()
    c.update(1, (15, -5))
    c.update(1, (15, 5))
    assert (c.count_in, c.count_out) == (0, 0)


def test_centroid_on_line_counted_once():
    c = make_counter()
    c.update(1, (5, -5))
    c.update(1, (5, 0))  # exactly on the line: cross product == 0
    c.update(1, (5, 5))
    assert (c.count_in, c.count_out) == (1, 0)


def test_zero_length_line_rejected():
    with pytest.raises(ValueError):
        LineCounter((5, 5), (5, 5))
