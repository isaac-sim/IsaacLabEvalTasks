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

import os
import numpy as np

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from io_utils import load_gr1_joints_config
from policies.image_conversion import resize_frames_with_padding
from policies.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from policies.policy_base import PolicyBase
from robot_joints import JointsAbsPosition

from isaaclab.sensors import Camera

from config.args import Gr00tN1ClosedLoopArguments
from gr00t.data.dataset import LeRobotSingleDataset


class GTPolicy(PolicyBase):
    def __init__(self, args: Gr00tN1ClosedLoopArguments):
        self.args = args
        self.policy = self._load_policy()
        self.policy_iter = iter(self.policy)
        self._load_policy_joints_config()
        self._load_sim_joints_config()

    def _load_policy_joints_config(self):
        """Load the policy joint config from the data config."""
        self.gr00t_joints_config = load_gr1_joints_config(self.args.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """Load the simulation joint config from the data config."""
        self.gr1_state_joints_config = load_gr1_joints_config(self.args.state_joints_config_path)
        self.gr1_action_joints_config = load_gr1_joints_config(self.args.action_joints_config_path)

    def _load_policy(self):
        """Load the policy from the model path."""
        assert os.path.exists(self.args.dataset_path), f"Dataset path {self.args.dataset_path} does not exist"

        # Use the same data preprocessor as the loaded fine-tuned ckpts
        self.data_config = DATA_CONFIG_MAP[self.args.data_config]

        modality_config = self.data_config.modality_config()

        return LeRobotSingleDataset(
            dataset_path=self.args.dataset_path,
            modality_configs=modality_config,
            video_backend=self.args.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=self.args.embodiment_tag,
        )

    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Call every simulation step to update policy's internal state."""
        pass

    def get_new_goal(
        self, current_state: JointsAbsPosition, ego_camera: Camera, language_instruction: str
    ) -> JointsAbsPosition:
        """
        Run policy prediction on the given observations. Produce a new action goal for the robot.

        Args:
            current_state: robot proprioceptive state observation
            ego_camera: camera sensor observation
            language_instruction: language instruction for the task

        Returns:
            A dictionary containing the inferred action for robot joints.
        """
        # data_point = self.policy.get_step_data(traj_id, step_count)
        data_point = next(self.policy_iter)
        actions = {
            "action.left_arm": np.array(data_point["action.left_arm"])[None, ...],
            "action.right_arm": np.array(data_point["action.right_arm"])[None, ...],
            "action.left_hand": np.array(data_point["action.left_hand"])[None, ...],
            "action.right_hand": np.array(data_point["action.right_hand"])[None, ...],
            "action.waist": np.array(data_point["action.waist"])[None, ...],
        }
        robot_action_sim = remap_policy_joints_to_sim_joints(
            actions, self.gr00t_joints_config, self.gr1_action_joints_config, self.args.simulation_device
        )
        return robot_action_sim

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        self.policy_iter = iter(self.policy)
        pass
