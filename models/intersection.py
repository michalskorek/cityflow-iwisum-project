from dataclasses import dataclass
from models.road_link import RoadLink
from models.traffic_light import TrafficLight
from models.point import Point
from typing import List

@dataclass
class Intersection:
    id: str
    point: Point
    width: int
    roads: List[str]
    roadLinks: List[RoadLink]
    trafficLight: TrafficLight
    virtual: bool