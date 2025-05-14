# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from abc import ABC, abstractmethod

from isaaclab.sensors import Camera

from robot_joints import JointsAbsPosition


class PolicyBase(ABC):
    """A base class for all policies."""

    @abstractmethod
    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Called every simulation step to update policy's internal state."""
        pass

    @abstractmethod
    def get_new_goal(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Generates a goal given the current state and camera observations."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the policy's internal state."""
        pass
