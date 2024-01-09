from dataclasses import dataclass
from models.lane_link import LaneLink
from typing import List


@dataclass
class RoadLink:
    type: str
    startRoad: str
    endRoad: str
    laneLinks: List[LaneLink]
