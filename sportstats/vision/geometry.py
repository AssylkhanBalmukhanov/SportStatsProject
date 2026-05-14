from __future__ import annotations

from math import sqrt
from typing import Sequence


BBox = Sequence[float]
Point = tuple[float, float]


def get_center_of_bbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def get_bbox_width(bbox: BBox) -> float:
    return float(bbox[2] - bbox[0])


def get_foot_position(bbox: BBox) -> Point:
    x1, _y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def measure_distance(point_1: Point, point_2: Point) -> float:
    return sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def measure_xy_distance(point_1: Point, point_2: Point) -> Point:
    return (point_1[0] - point_2[0], point_1[1] - point_2[1])


def bbox_to_int(bbox: BBox) -> tuple[int, int, int, int]:
    return tuple(int(round(value)) for value in bbox)  # type: ignore[return-value]
