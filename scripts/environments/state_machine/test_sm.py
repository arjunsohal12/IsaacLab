# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse

from torch import smm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence


from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.state_machine import *


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    obs_dict = env.reset()
    sm = StateMachine(env)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            sm.reset()
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            above_pos = object_position.clone()
            above_pos[:, 2] += 0.1
            object_position[:, 2] -= 0.05

            action_sequence = [
                PickObject(object="object", asset="robot", num_envs=env.num_envs, device=env.device),
            ]

            action_sequence_success, _ = sm.execute_action_sequence(action_sequence)
            print("action_sequence_success count:", action_sequence_success.sum().item())

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
