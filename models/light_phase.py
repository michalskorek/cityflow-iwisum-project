from dataclasses import dataclass
from typing import List

@dataclass
class LightPhase:
    time: int
    availableRoadLinks: List[int]