from dataclasses import dataclass
from models.light_phase import LightPhase
from typing import List

@dataclass
class TrafficLight:
    lightphases: List[LightPhase]