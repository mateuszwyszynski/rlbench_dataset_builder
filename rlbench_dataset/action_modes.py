import numpy as np
from abc import abstractmethod
from scipy.spatial.transform import Rotation as R


class ActionModeBase:
    @abstractmethod
    def convert_low_dim_obs_to_action(self, step: int, low_dim_obs: list[dict]):
        pass

    @abstractmethod
    def get_description(self):
        """Returns a brief description of the action."""
        pass

    @abstractmethod
    def get_shape(self):
        """Returns the shape the action."""
        pass


def get_next_gripper_open(step: int, low_dim_obs: list[dict]):
    if step == len(low_dim_obs) - 1:
        return low_dim_obs[step]["gripper_open"]
    else:
        return low_dim_obs[step + 1]["gripper_open"]


class JointVelocitiesActionMode(ActionModeBase):
    def __init__(self, joint_count):
        self.joint_count = joint_count

    def convert_low_dim_obs_to_action(self, step: int, low_dim_obs: list[dict]):
        next_gripper_open = get_next_gripper_open(step, low_dim_obs)
        return np.concatenate(
            [low_dim_obs[step]["joint_velocities"], [next_gripper_open]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return f"Action mode controlling the joint velocities ({self.joint_count}) and the discrete gripper state (1)."

    def get_shape(self):
        return (self.joint_count + 1,)


class AbsoluteJointPositionsActionMode(ActionModeBase):
    def __init__(self, joint_count):
        self.joint_count = joint_count

    def convert_low_dim_obs_to_action(self, step: int, low_dim_obs: list[dict]):
        if step == len(low_dim_obs) - 1:
            next_joint_position_action = low_dim_obs[step]["misc"][
                "joint_position_action"
            ]
        else:
            next_joint_position_action = low_dim_obs[step + 1]["misc"][
                "joint_position_action"
            ]
        return np.array(next_joint_position_action, dtype=np.float32)

    def get_description(self):
        return f"Action mode controlling the absolute joint positions ({self.joint_count}) and the discrete gripper state (1)"

    def get_shape(self):
        return (self.joint_count + 1,)


class DeltaJointPositionsActionMode(ActionModeBase):
    def __init__(self, joint_count):
        self.joint_count = joint_count

    def convert_low_dim_obs_to_action(self, step: int, low_dim_obs: list[dict]):
        if step == len(low_dim_obs) - 1:
            next_joint_position_action = low_dim_obs[step]["misc"][
                "joint_position_action"
            ]
        else:
            next_joint_position_action = low_dim_obs[step + 1]["misc"][
                "joint_position_action"
            ]

        delta_joint_positions = np.array(next_joint_position_action[:-1]) - np.array(
            low_dim_obs[step]["joint_positions"]
        )
        next_gripper_open = get_next_gripper_open(step, low_dim_obs)
        return np.concatenate(
            [delta_joint_positions, [next_gripper_open]], dtype=np.float32, axis=-1
        )

    def get_description(self):
        return f"Action mode controlling the delta joint positions ({self.joint_count}) and the gripper state (1)."

    def get_shape(self):
        return (self.joint_count + 1,)


class AbsoluteEEPoseActionMode(ActionModeBase):
    def convert_low_dim_obs_to_action(self, step: int, low_dim_obs: list[dict]):
        next_gripper_open = get_next_gripper_open(step, low_dim_obs)
        if step == len(low_dim_obs) - 1:
            return np.concatenate(
                [low_dim_obs[step]["gripper_pose"], [next_gripper_open]],
                axis=-1,
                dtype=np.float32,
            )
        else:
            return np.concatenate(
                [low_dim_obs[step + 1]["gripper_pose"], [next_gripper_open]],
                axis=-1,
                dtype=np.float32,
            )

    def get_description(self):
        return "Action mode controlling the absolute end-effector pose represented as x,y,z and quaternion (7) and the gripper state. (1)"

    def get_shape(self):
        return (8,)


class DeltaEEPoseActionMode(ActionModeBase):
    def convert_low_dim_obs_to_action(self, step: int, low_dim_obs: list[dict]):
        start_position = np.array(low_dim_obs[step]["gripper_pose"][:3])
        start_orientation = np.array(low_dim_obs[step]["gripper_pose"][3:])

        if step == len(low_dim_obs) - 1:
            end_position = start_position
            end_orientation = start_orientation
        else:
            end_position = np.array(low_dim_obs[step + 1]["gripper_pose"][:3])
            end_orientation = np.array(low_dim_obs[step + 1]["gripper_pose"][3:])

        delta_position = end_position - start_position
        start_rotation = R.from_quat(start_orientation)
        end_rotation = R.from_quat(end_orientation)
        delta_rotation = end_rotation * start_rotation.inv()
        delta_euler = delta_rotation.as_euler("xyz")

        next_gripper_open = get_next_gripper_open(step, low_dim_obs)
        return np.concatenate(
            [delta_position, delta_euler, [next_gripper_open]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return "Action mode controlling the delta end-effector pose (position x,y,z and orientation in Euler angles) (6) and the gripper state. (1)"

    def get_shape(self):
        return (7,)
