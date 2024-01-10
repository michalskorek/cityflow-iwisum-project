from dataclasses import dataclass
from models.intersection import Intersection
from models.road import Road
import json
from typing import List

from models.road_link import RoadLink


@dataclass
class Roadnet:
    intersections: List[Intersection]
    roads: List[Road]

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        intersections = [
            Intersection(**intersection_data)
            for intersection_data in data["intersections"]
        ]
        roads = [Road(**road_data) for road_data in data["roads"]]
        return cls(intersections=intersections, roads=roads)
