from models.extended_models.lane_extended import LaneExtended
from models.road import Road


class RoadExtended:
    def __init__(self, road: Road):
        self.id = road.id
        self.startIntersection = road.startIntersection
        self.endIntersection = road.endIntersection
        self.points = road.points
        self.lanes = []
        for i in range(len(road.lanes)):
            self.lanes.append(LaneExtended(road.lanes[i], f"{road.id}_{i}"))
