# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
