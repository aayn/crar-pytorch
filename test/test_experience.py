import numpy as np
import pytest


def compute_segment_indices(buffer, num_segments):
    if num_segments > len(buffer):
        raise ValueError("Number of segments > buffer size.")
    segment_indices = []
    idx = 0
    increment = len(buffer) // num_segments
    mod = len(buffer) % num_segments
    for i in range(num_segments):
        segment_indices.append(idx)
        idx += increment + int(i < mod)
    return segment_indices


def test_compute_segment_indices():
    expected = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    actual = compute_segment_indices(list(range(500)), 10)
    assert expected == actual

    expected = [0, 42, 84, 126, 168, 210, 252, 294, 336, 377, 418, 459]
    acutal = compute_segment_indices(list(range(500)), 12)
    assert expected == acutal

    with pytest.raises(ValueError, match=r"Number of segments"):
        compute_segment_indices(list(range(500)), 900)
