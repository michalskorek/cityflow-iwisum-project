import cityflow
from config import Config
from models.roadnet import Roadnet
from environment.cityflow_env import CityFlowEnv
import numpy as np
import random

if __name__ == "__main__":
    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    env = CityFlowEnv(roadnet, config)
    for i in range(5000):
        states_vect = env.get_states()
        actions = np.zeros(env.intersections_num, dtype=np.int8)
        for i, state_vect in enumerate(states_vect):
            actions[i] = random.choice(env.get_possible_actions(i))
        reward = env.step(actions)