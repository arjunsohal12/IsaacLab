from isaaclab.utils import configclass


@configclass
class CuroboControllerCfg:
    """Configuration for curobo inverse kinematics controller."""

    command_type = "pose"
    """Type of task-space command to control the articulation's body.

    If "position", then the controller only controls the position of the articulation's body.
    Otherwise, the controller controls the pose of the articulation's body.
    """

    use_relative_mode: bool = False
