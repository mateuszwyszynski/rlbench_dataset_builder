import numpy as np
from abc import abstractmethod
from scipy.spatial.transform import Rotation as R


class ProprioModeBase:
    @abstractmethod
    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        pass

    @abstractmethod
    def get_description(self):
        """Returns a brief description of the action."""
        pass

    @abstractmethod
    def get_shape(self):
        """Returns the shape the action."""
        pass


class JointPositionsProprioMode(ProprioModeBase):
    def __init__(self, joint_count):
        self.joint_count = joint_count

    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        return np.concatenate(
            [low_dim_obs[step]["joint_positions"], [low_dim_obs[step]["gripper_open"]]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return f"Joint positions angles ({self.joint_count}) and the discrete gripper state (1)."

    def get_shape(self):
        return (self.joint_count + 1,)


class AbsoluteEEQuaternionPoseProprioMode(ProprioModeBase):
    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        return np.concatenate(
            [low_dim_obs[step]["gripper_pose"], [low_dim_obs[step]["gripper_open"]]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return "The absolute end-effector pose represented as x,y,z and quaternion (7) and the gripper state. (1)"

    def get_shape(self):
        return (8,)


class AbsoluteEEEulerPoseProprioMode(ProprioModeBase):
    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        ee_position = low_dim_obs[step]["gripper_pose"][:3]
        ee_orientation = low_dim_obs[step]["gripper_pose"][3:]

        ee_rotation = R.from_quat(ee_orientation)
        ee_euler_orientation = ee_rotation.as_euler("xyz")

        return np.concatenate(
            [ee_position, ee_euler_orientation, [low_dim_obs[step]["gripper_open"]]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return "The absolute end-effector pose represented as x,y,z and euler angles (7) and the gripper state (1)."

    def get_shape(self):
        return (7,)
