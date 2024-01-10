from dataclasses import dataclass
from models.point import Point
from typing import List


@dataclass
class LaneLink:
    startLaneIndex: int
    endLaneIndex: int
    points: List[Point]
