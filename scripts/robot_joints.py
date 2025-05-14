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

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class JointsAbsPosition:
    joints_pos: torch.Tensor
    """Joint positions in radians"""

    joints_order_config: Dict[str, int]
    """Joints order configuration"""

    device: torch.device
    """Device to store the tensor on"""

    @staticmethod
    def zero(joint_order_config: Dict[str, int], device: torch.device):
        return JointsAbsPosition(joints_pos=torch.zeros((len(joint_order_config)), device=device),
                                 joints_order_config=joint_order_config,
                                 device=device)

    def to_array(self) -> torch.Tensor:
        return self.joints_pos.cpu().numpy()

    @staticmethod
    def from_array(array: np.ndarray, joint_order_config: Dict[str, int], device: torch.device) -> 'JointsAbsPosition':
        assert array.ndim == 1
        assert array.shape[0] == len(joint_order_config)
        return JointsAbsPosition(joints_pos=torch.from_numpy(array).to(device),
                                 joints_order_config=joint_order_config,
                                 device=device)

    def set_joints_pos(self, joints_pos: torch.Tensor):
        self.joints_pos = joints_pos.to(self.device)

    def get_joints_pos(self, device: torch.device = None) -> torch.Tensor:
        if device is None:
            return self.joints_pos
        else:
            return self.joints_pos.to(device)