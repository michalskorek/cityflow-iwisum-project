from models.lane import Lane


class LaneExtended:
    def __init__(self, lane: Lane, id: str):
        self.width = lane['width']
        self.maxSpeed = lane['maxSpeed']
        self.id = id