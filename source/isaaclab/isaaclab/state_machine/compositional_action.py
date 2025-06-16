from .actions import *
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from isaaclab.utils.math import subtract_frame_transforms

class CompositionalAction():

    def __init__(self, asset = "robot", num_envs : int = 1, device: torch.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_duration: int = 100):
        self.actions_list = []
        self.asset = asset
        self.num_envs = num_envs
        self.device = device,
        self.max_duration = max_duration
    def initialize(self, env):
         raise NotImplementedError

class PickObject(CompositionalAction):
    
    def __init__(self, object, asset = "robot", num_envs : int = 1, device: torch.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_duration: int = 100):
        super().__init__(asset, num_envs, device, max_duration)
        self.object = object

    def initialize(self, env):
        object_data = env.unwrapped.scene[self.object].data
        object_position = object_data.root_pos_w
        # will code these into the object itself later
        grasp_position = object_position.clone()
        grasp_position[:, 2] -= 0.05

        pre_grasp = object_position.clone()
        pre_grasp[:, 2] += 0.1
        
        self.actions_list = [
            Move(asset=self.asset, num_envs=env.num_envs, device=env.device, target_positions_w=pre_grasp, max_duration=self.max_duration),
            OpenGripper(asset=self.asset, num_envs=env.num_envs, device=env.device),
            Move(asset=self.asset, num_envs=env.num_envs, device=env.device, target_positions_w=grasp_position, max_duration=self.max_duration),
            CloseGripper(asset=self.asset, num_envs=env.num_envs, device=env.device),
            Move(asset=self.asset, num_envs=env.num_envs, device=env.device, target_positions_w=pre_grasp, max_duration=self.max_duration)
            ]
    def reset(self):
         for action in self.actions_list:
              action.reset()