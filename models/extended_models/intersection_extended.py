from typing import List
from models.extended_models.road_extended import RoadExtended
from models.intersection import Intersection
from models.road import Road


class IntersectionExtended:
    def __init__(self, intersection: Intersection, roads: List[Road]):
        self.id = intersection.id
        self.point = intersection.point
        self.width = intersection.width
        self.roadLinks = intersection.roadLinks
        self.trafficLight = intersection.trafficLight
        self.virtual = intersection.virtual
        self.roads = []
        for i in range(len(roads)):
            self.roads.append(RoadExtended(roads[i]))
