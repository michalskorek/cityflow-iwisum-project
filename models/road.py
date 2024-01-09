from dataclasses import dataclass
from models.point import Point
from models.lane import Lane
from typing import List


@dataclass
class Road:
    id: str
    startIntersection: str
    endIntersection: str
    points: List[Point]
    lanes: List[Lane]
