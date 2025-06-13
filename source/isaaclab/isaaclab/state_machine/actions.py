import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import omni.isaac.lab.utils.math as math_utils


class Action(ABC):
    def __init__(self, asset = "robot", num_envs : int = 1, device: torch.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_duration: int = 100):
        self.asset = asset
        self.num_envs = num_envs
        self.device = device
        self.max_duration = max_duration
        self.steps_taken = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
    def _convert_world_to_base_frame(self, env, env_ids: torch.Tensor, world_positions: torch.Tensor, world_orientations: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert world frame poses to robot base frame poses.
        
        Args:
            env: The environment to get transforms from
            env_ids: Tensor of environment indices
            world_positions: Positions in world frame, shape (len(env_ids), 3)
            world_orientations: Orientations in world frame, shape (len(env_ids), 4). If None, returns zero quaternions.
            
        Returns:
            Tuple of (base_positions, base_orientations) where:
            - base_positions: Positions in robot base frame, shape (len(env_ids), 3)
            - base_orientations: Orientations in robot base frame, shape (len(env_ids), 4)
        """
        env = env.unwrapped
        root_pose_w = env.scene["robot"].data.root_state_w[env_ids, :7]
        
        # Use identity quaternions if orientations not provided
        if world_orientations is None:
            world_orientations = torch.zeros((len(env_ids), 4), device=self.device)
            world_orientations[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0] (w, x, y, z)
        
        # Convert world poses to base frame using proper transform
        base_positions, base_orientations = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            world_positions, world_orientations
        )
        
        return base_positions, base_orientations
        
    @abstractmethod
    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the action values for environments that are not done.
        
        Args:
            env: The environment to get state from
            env_ids: Boolean mask indicating which environments are active
        
        Returns:
            Tuple of (action_values, done_mask) where:
            - action_values is the action tensor to apply
            - done_mask indicates which environments completed this action
        """
        pass

    def compute_action(self, env, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update steps and check timeout
        # TODO: Fail and finish at the same time should probably be a success (we already did the work, might as well mark it as success)
        self.steps_taken[env_ids] += 1
        timeout_failure = self.steps_taken >= self.max_duration
            
        action, success = self._compute_action_impl(env, env_ids)
        return action, success, timeout_failure
    
    def reset(self):
        """Reset the action's internal state"""
        self.steps_taken = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)


class Move(Action):
    def __init__(self, num_envs: int, device: torch.device, target_positions_w: torch.Tensor, max_duration: int = 100):
        """
        Args:
            num_envs: Number of parallel environments
            device: Device to use for tensors
            target_positions: Tensor of shape (num_envs, 3) containing target positions
        """
        super().__init__(num_envs, device, max_duration)

        self.target_positions_w = target_positions_w
        self.position_threshold = 0.01  # Distance threshold to consider target reached
        self.gripper_threshold = 0.001
        
    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert target positions from world frame to robot base frame
        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, self.target_positions_w[env_ids])

        # Create action tensor
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b
        
        # Keep orientation fixed (pointing down)
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0


        # We need to send some action to the gripper
        # Our goal is not to have it do anything
        # If it is fully open, we keep sending the open command
        # In any other case if it is static it will the fully closed or holding onto something
        # (as it can only be controlled by binary action)
        # Therefore, we keep sending the close command
        # We are doing this to avoid any memory in the system (remembering the last action)
        gripper_pos = env.scene["robot"].data.joint_pos[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        current_pos = env.unwrapped.obs_dict["ee_pos"]
        # TODO: Check if both of these are in full world and not env world frame
        distance = torch.norm(self.target_positions_w[env_ids] - current_pos[env_ids], dim=1)
        new_done = distance < self.position_threshold
        
        return action, new_done


class GripperAction(Action):
    def __init__(self, num_envs: int, device: torch.device, target_value: float, duration: int = 20):
        super().__init__(num_envs, device)
        self.target_value = target_value
        self.duration = duration
        
    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create action tensor
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        
        # Get current end effector position in world frame and convert to base frame
        env = env.unwrapped
        ee_pos_w = env.scene["ee_frame"].data.target_pos_w[env_ids, 0, :] - env.scene.env_origins[env_ids]
        ee_pos_b, _ = self._convert_world_to_base_frame(env, env_ids, ee_pos_w)

        action[:, 0:3] = ee_pos_b
        
        # Keep orientation fixed (pointing down)
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        action[:, 7] = self.target_value
        
        new_done = self.steps_taken[env_ids] >= self.duration
        
        return action, new_done


class OpenGripper(GripperAction):
    def __init__(self, num_envs: int, device: torch.device):
        super().__init__(num_envs, device, target_value=1.0)


class CloseGripper(GripperAction):
    def __init__(self, num_envs: int, device: torch.device):
        super().__init__(num_envs, device, target_value=-1.0)


class ParallelHighLevelEnv:
    def __init__(self, env):
        """
        Args:
            env: The underlying environment that supports parallel execution
        """
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.unwrapped.device

        # State machine tracking
        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = []
        self.all_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Observation history tracking
        self.obs_history = None  # Will be initialized in set_action_sequence
        self.frame_history = []
    
    def reset(self):
        """Reset the environment and action state"""
        # TODO: Add independent resets (would need to update actions with specific new positions usually)
        #       Maybe action can have a function attached, so that it calculates in real time (like obs terms)
        obs = self.env.reset()[0]
        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_sequence_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_sequence_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.combined_action = torch.zeros((self.num_envs, 8), device=self.device)
        
        # Reset all actions
        for action in self.actions:
            action.reset()
            
        return obs
    
    def set_action_sequence(self, actions):
        """
        Set the sequence of actions to execute.
        
        Args:
            actions: List of Action objects to execute in sequence
        """
        self.actions = actions
        # Reset all actions
        for action in self.actions:
            action.reset()
            
        # Initialize observation history tensors
        self.obs_history = {}
        for k, v in self.env.unwrapped.obs_dict.items():
            # Shape: (num_actions + 1, num_envs, K) - +1 for initial state
            self.obs_history[k] = torch.zeros((len(actions) + 1, self.num_envs, v.shape[1]), device=self.device)
            self.obs_history[k][0] = v  # Store initial observations

        self.frame_history = []

    def _get_obs_summary(self) -> str:
        """Generate a human-readable summary of observation changes.
        
        Returns:
            str: Formatted string showing observation changes between actions
        """
        summary = "Initial observations:\n"
        precision = 3  # Number of decimal places to show and use for thresholding
        threshold = 10 ** (-precision)
        
        # Add initial observations
        for k, v in self.obs_history.items():
            summary += f"  {k}: {v[0, 0].cpu().numpy().round(precision)}\n"
            
        # Add changes after each action
        for action_idx in range(len(self.actions)):
            # Check if any env completed this action
            if not (self.current_action_idx > action_idx).any():
                continue
                
            summary += f"\nChanges after action {action_idx + 1}:\n"
            for k, v in self.obs_history.items():
                prev_vals = v[action_idx, 0]
                curr_vals = v[action_idx + 1, 0]
                if not (torch.abs(prev_vals - curr_vals) < threshold).all():
                    summary += f"    {k}: {curr_vals.cpu().numpy().round(precision)}\n"
                    
        return summary

    def execute_action_sequence(self, actions) -> Tuple[torch.Tensor, str]:
        """Execute a sequence of actions and return success status and observation summary.
        
        Args:
            actions: List of Action objects to execute in sequence
            
        Returns:
            Tuple containing:
            - torch.Tensor: Boolean tensor indicating which environments succeeded
            - str: Summary of observation changes during execution
        """
        self.set_action_sequence(actions)
        while not (self.action_sequence_success | self.action_sequence_failure).all():
            self.step()
            if len(self.frame_history) == 0:
                self.frame_history.append(self.env.video_recorder.recorded_frames[-1])
        obs_summary = self._get_obs_summary()

        # for frame in self.frame_history:
        #     plt.imshow(frame)
        #     plt.show()
        # print("obs_summary", obs_summary)
        return self.action_sequence_success, obs_summary, self.frame_history
    
    def step(self):
        """
        Execute one step of the current action for all environments.
        Environments that complete their current action will move to the next action.
        
        Returns:
            observation: The current observation
            done: Boolean tensor indicating which environments have completed all actions
        """
        # Create mask for environments that are still active
        active_mask = ~(self.action_sequence_success | self.action_sequence_failure)
        
        # TODO: Still running loop now, even if everything is done
        # if not active_mask.any():
        #     return None, self.all_done
            
        # Get all unique action indices that are currently active
        unique_action_indices = torch.unique(self.current_action_idx[active_mask])
        # print("unique_action_indices", unique_action_indices)
        
        # Initialize action tensor
        # TODO: Slightly inelegant, for envs that are done, we want to keep sending the last action to stay in place
        # combined_action = torch.zeros((self.num_envs, 8), device=self.device)
        combined_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Compute actions for each unique action type
        for action_idx in unique_action_indices:
            # Get mask for environments executing this action
            action_mask = (self.current_action_idx == action_idx) & active_mask
            # Convert boolean mask to indices
            action_env_ids = torch.nonzero(action_mask).squeeze(-1)
                
            # Get the action and compute its values
            action = self.actions[action_idx]
            action_values, action_success, action_failure = action.compute_action(self.env, action_env_ids)
            
            # Update combined action and done masks
            self.combined_action[action_env_ids] = action_values
            combined_success[action_env_ids] = action_success

            # Store observations for environments that completed this action
            if action_success.any():
                completed_envs = torch.nonzero(action_success).squeeze(-1)
                for k, v in self.env.unwrapped.obs_dict.items():
                    self.obs_history[k][action_idx + 1, completed_envs] = v[completed_envs]
                if 0 in completed_envs:
                    self.frame_history.append(self.env.video_recorder.recorded_frames[-1])

            self.action_sequence_failure = self.action_sequence_failure | action_failure

        # print("unique_action_indices", unique_action_indices)
        # print("gripper pos", self.env.unwrapped.scene["robot"].data.joint_pos[:, -1])
        
        # Step the environment ONCE with the combined actions
        obs = self.env.step(self.combined_action)[0]
        
        # Update action indices only for environments that completed their current action
        self.current_action_idx[combined_success] += 1
        
        # Mark environments as done if they've completed all actions
        self.action_sequence_success = self.action_sequence_success | (self.current_action_idx >= len(self.actions))
        
        return obs, self.action_sequence_success
    
    def close(self):
        self.env.close()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped 