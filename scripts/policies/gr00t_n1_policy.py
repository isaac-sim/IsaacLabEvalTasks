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
from typing import Dict, List

import numpy as np
import torch

from isaaclab.sensors import Camera

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from config.args import Gr00tN1ClosedLoopArguments
from policies.image_conversion import resize_frames_with_padding
from policies.policy_base import PolicyBase
from robot_joints import JointsAbsPosition
from io_utils import load_gr1_joints_config


def remap_sim_joints_to_policy_joints(sim_joints_state: JointsAbsPosition,
                                      policy_joints_config: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Remap the state or actions joints from simulation joint orders to policy joint orders
    """
    data = {}
    assert isinstance(sim_joints_state, JointsAbsPosition)
    for group, joints_list in policy_joints_config.items():
        data[group] = []
        for joint_name in joints_list:
            if joint_name in sim_joints_state.joints_order_config.keys():
                joint_index = sim_joints_state.joints_order_config[joint_name]
                data[group].append(sim_joints_state.joints_pos[:, joint_index])
            else:
                raise ValueError(f"Joint {joint_name} not found in {sim_joints_state.joints_order_config}")

        data[group] = np.stack(data[group], axis=1)
    return data


def remap_policy_joints_to_sim_joints(policy_joints: Dict[str, np.array],
                                      policy_joints_config: Dict[str, List[str]],
                                      sim_joints_config: Dict[str, int],
                                      device: torch.device) -> JointsAbsPosition:
    """
    Remap the actions joints from policy joint orders to simulation joint orders
    """
    # assert all values in policy_joint keys are the same shape and save the shape to init data
    policy_joint_shape = None
    for _, joint_pos in policy_joints.items():
        if policy_joint_shape is None:
            policy_joint_shape = joint_pos.shape
        else:
            assert joint_pos.ndim == 3
            assert joint_pos.shape[:2] == policy_joint_shape[:2]

    assert policy_joint_shape is not None
    data = torch.zeros([policy_joint_shape[0], policy_joint_shape[1], len(sim_joints_config)],
                       device=device)
    for joint_name, gr1_index in sim_joints_config.items():
        match joint_name.split("_")[0]:
            case "left":
                joint_group = "left_arm"
            case "right":
                joint_group = "right_arm"
            case "L":
                joint_group = "left_hand"
            case "R":
                joint_group = "right_hand"
            case _:
                continue
        if joint_name in policy_joints_config[joint_group]:
            gr00t_index = policy_joints_config[joint_group].index(joint_name)
            data[..., gr1_index] = torch.from_numpy(
                policy_joints[f'action.{joint_group}'][..., gr00t_index]).to(device)

    sim_joints = JointsAbsPosition(joints_pos=data, joints_order_config=sim_joints_config, device=device)
    return sim_joints


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

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        # load the policy
        return Gr00tPolicy(
            model_path=self.args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.args.embodiment_tag,
            denoising_steps=self.args.denoising_steps,
            device=self.args.policy_device
        )

    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Call every simulation step to update policy's internal state."""
        pass

    def get_new_goal(self, current_state: JointsAbsPosition, ego_camera: Camera, language_instruction: str) -> JointsAbsPosition:
        """
        Run policy prediction on the given observations. Produce a new action goal for the robot.

        Args:
            current_state: robot proprioceptive state observation
            ego_camera: camera sensor observation
            language_instruction: language instruction for the task

        Returns:
            A dictionary containing the inferred action for robot joints.
        """
        rgb = ego_camera.data.output['rgb']
        # Apply preprocessing to rgb
        rgb = resize_frames_with_padding(rgb, target_image_size=self.args.target_image_size, bgr_conversion=False, pad_img=True)
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
        }
        robot_action_policy = self.policy.get_action(observations)

        robot_action_sim = remap_policy_joints_to_sim_joints(robot_action_policy,
                                                             self.gr00t_joints_config,
                                                             self.gr1_action_joints_config,
                                                             self.args.simulation_device)

        return robot_action_sim

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        pass
