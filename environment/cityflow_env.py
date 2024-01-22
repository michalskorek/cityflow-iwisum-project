from collections import defaultdict
from typing import List
from config import Config
from models.extended_models.intersection_extended import IntersectionExtended
from models.road import Road
from models.roadnet import Roadnet
import numpy as np
import cityflow


class CityFlowEnv:
    def __init__(self, roadnet: Roadnet, config: Config):
        self.intersections = list(
            filter(lambda x: not x.virtual, self.build_extended_intersections(roadnet))
        )
        self.lane_phase_dict = self.build_lane_phase_dict()
        self.intersections_num = len(self.intersections)
        self.cityflow_engine = cityflow.Engine(config.cityflow_config_path)
        self.last_lane_waiting_vehicles = (
            self.cityflow_engine.get_lane_waiting_vehicle_count()
        )

    def build_extended_intersections(
        self, roadnet: Roadnet
    ) -> List[IntersectionExtended]:
        intersection_extended_list = []
        for intersection in roadnet.intersections:
            if intersection.virtual:
                continue
            intersection_roads = [
                road for road in roadnet.roads if road.id in intersection.roads
            ]
            intersection_extended_list.append(
                IntersectionExtended(intersection, intersection_roads)
            )
        return intersection_extended_list

    def build_lane_phase_dict(self) -> dict:
        lane_phase_dict = defaultdict(set)
        for i, intersection in enumerate(self.intersections):
            for j, lightPhase in enumerate(intersection.trafficLight["lightphases"]):
                for k, availableRoadLink in enumerate(lightPhase["availableRoadLinks"]):
                    roadLink = intersection.roadLinks[availableRoadLink]
                    startRoadId = roadLink["startRoad"]
                    for l, road in enumerate(intersection.roads):
                        if road.id == startRoadId:
                            startRoadIndex = l
                            break
                    for l, laneLink in enumerate(roadLink["laneLinks"]):
                        lane_phase_dict[
                            (i, startRoadIndex, laneLink["startLaneIndex"])
                        ].add(j)
        return lane_phase_dict

    def random_step(self):
        for i, intersection in enumerate(self.intersections):
            possible_actions = self.get_possible_actions(i)
            self.cityflow_engine.set_tl_phase(
                intersection.id, np.random.choice(possible_actions)
            )
        self.cityflow_engine.next_step()

    def step(self, actions):
        for intersection, phase_id in zip(self.intersections, actions):
            self.cityflow_engine.set_tl_phase(intersection.id, phase_id)
        self.cityflow_engine.next_step()
        new_lane_waiting_vehicles = (
            self.cityflow_engine.get_lane_waiting_vehicle_count()
        )

        reward = self.get_reward(new_lane_waiting_vehicles)
        self.last_lane_waiting_vehicles = new_lane_waiting_vehicles
        return reward

    def get_states(self):
        lane_waiting_vehicles = self.cityflow_engine.get_lane_waiting_vehicle_count()
        state = np.array(
            [
                np.zeros(len(intersection.trafficLight["lightphases"]))
                for intersection in self.intersections
            ]
        )
        for i, intersection in enumerate(self.intersections):
            intersection_phases_number = len(intersection.trafficLight["lightphases"])
            for j, road in enumerate(intersection.roads):
                for k, lane in enumerate(lane_waiting_vehicles):
                    phases = self.lane_phase_dict[(i, j, k)]
                    if len(phases) == intersection_phases_number:
                        continue
                    for phase in phases:
                        state[i][phase] += lane_waiting_vehicles[lane]
        return state

    def get_possible_actions(self, intersection_index: int):
        return list(
            range(
                len(self.intersections[intersection_index].trafficLight["lightphases"])
            )
        )

    def get_reward(self, new_lane_waiting_vehicles) -> List[int]:
        rewards = [
            -np.sum(
                new_lane_waiting_vehicles[lane.id]
                for road in intersection.roads
                for lane in road.lanes
            )
            for intersection in self.intersections
        ]
        return rewards
