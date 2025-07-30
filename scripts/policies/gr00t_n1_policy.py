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

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from io_utils import load_gr1_joints_config
from policies.image_conversion import resize_frames_with_padding
from policies.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from policies.policy_base import PolicyBase
from robot_joints import JointsAbsPosition

from isaaclab.sensors import Camera

from config.args import Gr00tN1ClosedLoopArguments


class Gr00tN1Policy(PolicyBase):
    def __init__(self, args: Gr00tN1ClosedLoopArguments):
        self.args = args
        self.policy = self._load_policy()
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
        assert os.path.exists(self.args.model_path), f"Model path {self.args.model_path} does not exist"

        # Use the same data preprocessor as the loaded fine-tuned ckpts
        self.data_config = DATA_CONFIG_MAP[self.args.data_config]
        # Add waist to the action keys
        self.data_config.action_keys.append("action.waist")

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        # load the policy
        return Gr00tPolicy(
            model_path=self.args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.args.embodiment_tag,
            denoising_steps=self.args.denoising_steps,
            device=self.args.policy_device,
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
        rgb = ego_camera.data.output["rgb"]
        # Apply preprocessing to rgb
        rgb = resize_frames_with_padding(
            rgb, target_image_size=self.args.target_image_size, bgr_conversion=False, pad_img=True
        )
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        robot_state_policy = remap_sim_joints_to_policy_joints(current_state, self.gr00t_joints_config)

        # Pack inputs to dictionary and run the inference
        observations = {
            "annotation.human.action.task_description": [language_instruction],  # list of strings
            "video.ego_view": rgb.reshape(-1, 1, 256, 256, 3),  # numpy array of shape (N, 1, 256, 256, 3)
            "state.left_arm": robot_state_policy["left_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.right_arm": robot_state_policy["right_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.left_hand": robot_state_policy["left_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
            "state.right_hand": robot_state_policy["right_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
            "state.waist": robot_state_policy["waist"].reshape(-1, 1, 1),  # numpy array of shape (N, 1, 1)
        }
        robot_action_policy = self.policy.get_action(observations)

        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.gr00t_joints_config, self.gr1_action_joints_config, self.args.simulation_device
        )

        return robot_action_sim

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        pass
