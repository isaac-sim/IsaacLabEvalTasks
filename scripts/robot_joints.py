# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from dataclasses import dataclass
from typing import Dict, List

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