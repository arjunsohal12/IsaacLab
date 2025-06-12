# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.observations import joint_pos
import torch
from typing import TYPE_CHECKING
import gym
from isaaclab.utils.math import apply_delta_pose, compute_pose_error
# from isaaclab.envs import ManagerBasedRLEnv
# from omni.isaac.core.utils.types import ArticulationAction
import carb
from pxr import UsdGeom
from isaacsim.core.api.objects import sphere, cuboid

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper, get_mesh_attrs, Mesh, get_cube_attrs, get_capsule_attrs, get_cylinder_attrs, get_sphere_attrs
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
import numpy as np
from pxr import UsdGeom, Usd, Gf
from pxr.UsdGeom import XformCache
if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg

def decompose_matrix(matrix: Gf.Matrix4d):
    """
    Converts a Gf.Matrix4d to (position, quaternion) as numpy arrays.
    """
    transform = Gf.Transform()
    transform.SetMatrix(matrix)
    translation = transform.GetTranslation()
    rotation = transform.GetRotation().GetQuat()

    pos = [translation[0], translation[1], translation[2]]
    quat = [rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2], rotation.GetReal()]
    return pos, quat

class CuroboFrankaController:
    r"""Curobocontroller.

    """

    def __init__(self, cfg, env: gym.Env, num_envs: int, agent_name, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

        self.agent_name = agent_name
        self.env = env
        self.usd_helper = UsdHelper()
        self.robot = self.env.scene._articulations[self.agent_name] # maybe try and find better way to get this
        self.dof_names = self.robot.joint_names

        # combine scene dicts to get an objects dict, try to change this to be cleaner later, not sure how to handle it now
        # maybe add self.env.scene._extras for prims, not sure if we need to update prim positions since they dont do much
        self.objects = {**self.env.scene._articulations,  **self.env.scene._rigid_objects}
        # Define the joint names for the robot, not sure why this is here when we access above? Check it
        self.cmd_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.cmd_plan = None
        self.cmd_idx = 0
        self.idx_list = None

        self._num_actions = self.robot.num_joints
    

        self.setup_robot_model()

        self.setup_world_model()

        self.setup_motion_generation()

        self.pos_offset = (0.0, 0.0, 0.1034)
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    def set_command(
        self, command: torch.Tensor):
        # Apply transformation to self.command from robot finger to robot hand
        self._command = command
        self._command[:, 0:3] += torch.as_tensor(self.pos_offset, device=self._command.device)
        return None

    def setup_world_model(self) -> None:
        """
        Sets up the world model for CuRobo
        """
        # Create config
        self._world_cfg = WorldConfig()
        # list of prims and xforms to ignore when generating obstacles
        self.ignore_substring = ["Looks", "Materials", "defaultGroundPlane", "Visuals", "physicsScene", "collisions"]
        # Load stage and retrieve potential obstacles
        self.usd_helper.load_stage(self.env.sim.stage)
        obstacles = self.usd_helper.get_obstacles_from_stage(ignore_substring=self.ignore_substring).get_collision_check_world()
        # Add obstacles to our world model config
        for obstacle in obstacles:
            self._world_cfg.add_obstacle(obstacle)
        print(obstacles)
        # prims go under extras, lets see if we can load prims and rigid objects directly from the scene into the obstacles, 
        # Each instanceable object will have meshes somewhere under it, in a child xform. See if we can extract those for the faces/vertices of world model
        # metadata = get_mesh_attrs(self.env.scene.rigid_objects["object"])
        self.add_instanceable_object_from_path("/World/envs/env_0/Table/Collisions/Cube", obj_name="table")

        self.add_instanceable_object_from_path("/World/envs/env_0/Object/collisions/collisions", obj_name="object")

        print(self._world_cfg.objects)
        print(self.objects)
        file_path = "/home/arjun/Desktop/debug_mesh.obj"
        print("saving the world")
        self._world_cfg.save_world_as_mesh(file_path)

        # print(f)
        # Create the world model
        self.obstacle_map = {obstacle.name: obstacle for obstacle in obstacles}

    def add_instanceable_object_from_path(self, path, obj_name, prim=True):
        mesh_prim = self.env.sim.stage.GetPrimAtPath(path)
        
        if not prim:
            rigidbody = self.env.scene._rigid_objects[obj_name]
            position, orientation = rigidbody.data.root_state_w[0, :3], rigidbody.data.root_state_w[0, 3:7]
        else:
            position, orientation = self.get_world_pose_from_any_prim(mesh_prim)

        if mesh_prim.IsInstance():
            # mesh_prim is the “root” of an mesh_primance
            mesh_prim = mesh_prim.GetPrototype()
        elif mesh_prim.IsInstanceProxy():
            # mesh_prim is a proxy under one of the mesh_primances
            mesh_prim = mesh_prim.GetPrimInPrototype()
        else:
            # not mesh_primanced at all
            mesh_prim = mesh_prim

        print(mesh_prim.GetTypeName())
        print("  Attributes:", [a.GetName() for a in mesh_prim.GetAttributes()])
        print(mesh_prim.IsA(UsdGeom.Mesh))

        if mesh_prim.IsA(UsdGeom.Cube):
            metadata = get_cube_attrs(mesh_prim, cache=self.usd_helper._xform_cache)
        elif mesh_prim.IsA(UsdGeom.Sphere):
            metadata = get_sphere_attrs(mesh_prim, cache=self.usd_helper._xform_cache)
        elif mesh_prim.IsA(UsdGeom.Mesh):
            metadata = get_mesh_attrs(mesh_prim, cache=self.usd_helper._xform_cache)
            if metadata is None: # Curobo only support Triangle mesh, treat as cube for now until I find a better solution
                metadata = get_cube_attrs(mesh_prim, cache=self.usd_helper._xform_cache)
        elif mesh_prim.IsA(UsdGeom.Cylinder):
            metadata = get_cylinder_attrs(mesh_prim, cache=self.usd_helper._xform_cache)
        elif mesh_prim.IsA(UsdGeom.Capsule):
            metadata = get_capsule_attrs(mesh_prim, cache=self.usd_helper._xform_cache)

        object_cfg = getattr(self.env.cfg.scene, obj_name)
        metadata.pose = list(position) + object_cfg.init_state.rot
        metadata.name = obj_name
        print("Local Position:", position)
        print("Local Rotation:", orientation)
        print(metadata)
        # print(f)
        self._world_cfg.add_obstacle(metadata)

    def get_world_pose_from_any_prim(self, prim, time=Usd.TimeCode.Default()):
        # If it's a proxy under an instance, get the actual instance path
        xform_cache = XformCache(time)
        world_matrix = xform_cache.GetLocalToWorldTransform(prim)

        transform = Gf.Transform()
        transform.SetMatrix(world_matrix)

        pos = transform.GetTranslation()
        quat = transform.GetRotation().GetQuat()

        return (
            [pos[0], pos[1], pos[2]],
            [quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2], quat.GetReal()]
        )
        
    def setup_motion_generation(self) -> None:
        """
        Sets up motion generator for CuRobo
        """
        # Motion Generator config
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self._world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.02,
            collision_cache={"obb": 20, "mesh": 20},
            store_ik_debug=False,
            store_trajopt_debug=False,
            velocity_scale=0.6,
        )
        # Create Motion Generator
        self.motion_gen = MotionGen(self.motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(parallel_finetune=True)
        pose_metric = None
        # Create config for motion generation plans
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            max_attempts=10,
            enable_graph_attempt=3,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=pose_metric,
            time_dilation_factor=0.75
        )
    def setup_robot_model(self) -> None:
        """
        Sets up CuRobo robot model
        """
        # Load pre-created config from yaml as a dict
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg["kinematics"]["base_link"] = "panda_link0"  # setup robot base frame
        self.robot_cfg["kinematics"]["ee_link"] = "panda_hand"  # setup EE frame
        # collision spheres
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        self.robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"

        self.tensor_args = TensorDeviceType()
        # create CuRobo config from our dict
        self.robot_cfg_kinematics = RobotConfig.from_dict(self.robot_cfg, self.tensor_args)
        # Setup CuRobo model from config
        self.kinematics_model = CudaRobotModel(self.robot_cfg_kinematics.kinematics)


    def plan(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        cu_js: JointState,
    ) -> MotionGenResult:
        '''This function generates a motion plan for a robot's end-effector to reach a specified translation 
        and orientation goal. The function takes in the desired end-effector position and orientation, along with the current joint 
        state of the robot, and uses inverse kinematics to compute a feasible trajectory. This is done using a motion generation 
        system configured previously, with the resulting plan returned as a MotionGenResult.'''
        print("PLANNING:")
        print(ee_translation_goal)
        print(ee_orientation_goal)
        
        # Define the goal pose
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )

        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        self.save_counter = 0

        return result
    

    def forward(self, cu_js: JointState) -> torch.Tensor:
        '''This function executes a motion plan by computing and returning the appropriate 
        joint actions to move a robot's end-effector towards a target position and orientation. 
        If a plan is not already available, the function generates one using the plan method. It 
        then executes the plan by incrementally updating the joint states, which are returned as an 
        ArticulationAction object.'''

        if self.cmd_plan is None:
            self.cmd_idx = 0
            # Set EE goals
            ee_translation_goal = self._command[:, 0:3]
            ee_orientation_goal = self._command[:, 3:7]
            # compute curobo solution:
            result = self.plan(ee_translation_goal, ee_orientation_goal, cu_js)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                self.idx_list = [i for i in range(len(self.cmd_js_names))]
                self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_js_names)
                print("Path Planning Success")
            else:
                carb.log_warn("Plan did not converge to a solution.")
                return None



        cmd_state = self.cmd_plan[self.cmd_idx]
        self.cmd_idx += 1

        # once we have finished current command, set it to none to create a new motion plan
        if self.cmd_idx >= len(self.cmd_plan.position):
            self.cmd_idx = 0
            self.cmd_plan = None
        
        return cmd_state.position
    

    def compute(self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        ) -> torch.Tensor | None:
        """
        This function calculates the joint positions needed to achieve the desired end-effector pose. It periodically 
        updates internal parameters and uses the forward method to generate a motion command based on the current joint state. 
        If a valid command is generated, it adjusts for any offsets and returns the updated joint positions.
        """

        import time
        t0 = time.time()

        for object in self._world_cfg.objects:
            if object.name in self.objects:
                object.pose[:3] = self.objects[object.name].data.root_state_w[0, :3]
            # self.motion_gen.world_model.update_obstacle_pose(self.obstacle_map[obstacle_name].pose, name=obstacle_name) Figure out how to update this

        
        joint_positions = joint_positions.squeeze(0)
        joint_velocities = joint_velocities.squeeze(0)

        cu_js = JointState(
            position=self.tensor_args.to_device(joint_positions),
            velocity=self.tensor_args.to_device(joint_velocities),
            acceleration=self.tensor_args.to_device(joint_velocities) * 0.0,
            jerk=self.tensor_args.to_device(joint_velocities) * 0.0,
            joint_names=self.dof_names,
        )

        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        
        self.ee_pose = self.motion_gen.kinematics.compute_kinematics(cu_js).ee_pose
        print(self.ee_pose)

        art_action = self.forward(cu_js)



        obstacle_names = [obj.name for obj in self._world_cfg.objects if hasattr(obj, "name")]
        obstacle_info = {obj.name: obj.pose for obj in self._world_cfg.objects if hasattr(obj, 'pose') and hasattr(obj, 'name')}
        print(self._world_cfg.objects)
        print(obstacle_info)
        pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        # sph_list = self.kinematics_model.get_robot_as_spheres(joint_positions)
        # for si, s in enumerate(sph_list[0]):
        #     sp = sphere.VisualSphere(
        #         prim_path="/curobo/robot_sphere_" + "_" + str(si),
        #         position=np.ravel(s.position)
        #         + pose.position[0].cpu().numpy(),
        #         radius=float(s.radius),
        #         color=np.array([0, 0.8, 0.2]),
        #     )
        for i, object in enumerate(self._world_cfg.objects):
            cube = cuboid.VisualCuboid(
                prim_path = "/curobo/object" + "_" + str(i),
                position = object.pose[:3],
                orientation = object.pose[3:7],
                scale = object.dims
            )
        # cub_list = self._world_cfg.get_mesh_world()

        if art_action is not None:

            t1 = time.time()
            print("forwarding time:", t1-t0)            
            return art_action
        

    def attach_obj(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        js_names: list,
        asset_name: str,
    ) -> None:
        '''This function attaches an object to the robot as a part of the grasping skill. The object is attached as
        external links to the robot, we need to make sure to call this each time we grasp an object so the internal representation moves with us
        we dont currently use it
        '''

        # self.update()

        for item in self._world_cfg:
            if asset_name in item.name:
                prim_path = item.name

        cu_js = JointState(
            position=self.tensor_args.to_device(joint_positions),
            velocity=self.tensor_args.to_device(joint_positions) * 0.0,
            acceleration=self.tensor_args.to_device(joint_velocities) * 0.0,
            jerk=self.tensor_args.to_device(joint_velocities) * 0.0,
            joint_names=js_names,
        )

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [prim_path],
            surface_sphere_radius = 0.01,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_INSIDE,
            # world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0], self.tensor_args),
        )