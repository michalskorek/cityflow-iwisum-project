import json
import os

class Config:
    def __init__(self):
        self.cityflow_config_path = os.path.join("cityflow-config", "config.json")
        with open(self.cityflow_config_path) as config:
            self.config = json.load(config)
        self.roadnet_path = os.path.join(self.config["dir"], self.config["roadnetFile"])