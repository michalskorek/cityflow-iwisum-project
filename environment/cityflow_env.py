from typing import List
from config import Config
from models.extended_models.intersection_extended import IntersectionExtended
from models.road import Road
from models.roadnet import Roadnet
import numpy as np
import cityflow

class CityFlowEnv:
    def __init__(self, roadnet: Roadnet, config: Config):
        self.intersections = list(filter(lambda x: not x.virtual, self.build_extended_intersections(roadnet)))
        self.intersections_num = len(self.intersections)
        self.cityflow_engine = cityflow.Engine(config.cityflow_config_path)
        self.last_lane_waiting_vehicles = self.cityflow_engine.get_lane_waiting_vehicle_count()
    
    def build_extended_intersections(self, roadnet: Roadnet) -> List[IntersectionExtended]:
        intersection_extended_list = []
        for intersection in roadnet.intersections:
            if intersection.virtual:
                continue
            intersection_roads = [road for road in roadnet.roads if road.id in intersection.roads]
            intersection_extended_list.append(IntersectionExtended(intersection, intersection_roads))
        return intersection_extended_list
     
    def step(self, actions):
        for intersection, phase_id in zip(self.intersections, actions):
            self.cityflow_engine.set_tl_phase(intersection.id, phase_id)
        self.cityflow_engine.next_step()
        new_lane_waiting_vehicles = self.cityflow_engine.get_lane_waiting_vehicle_count()

        reward = self.get_reward(new_lane_waiting_vehicles)
        self.last_lane_waiting_vehicles = new_lane_waiting_vehicles
        return reward
    
    def get_states(self):
        lane_waiting_vehicles = self.cityflow_engine.get_lane_waiting_vehicle_count()
        intersections_state_vector = [
            [
                lane_waiting_vehicles[lane.id] for road in intersection.roads for lane in road.lanes
            ]
            for intersection in self.intersections
        ]

        return intersections_state_vector
    
    def get_possible_actions(self, intersection_index: int):
        return list(range(len(self.intersections[intersection_index].trafficLight["lightphases"])))
    
    def get_reward(self, new_lane_waiting_vehicles) -> List[int]:
        rewards = [
            np.sum(
                self.last_lane_waiting_vehicles[lane.id] - new_lane_waiting_vehicles[lane.id]
                for road in intersection.roads for lane in road.lanes
            )
            for intersection in self.intersections
        ]
        return rewards
