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

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import (
    ExhaustPipeGR1T2BaseEnvCfg,
)

joint_names_dict = {
    # arm joint
    "left_shoulder_pitch_joint": 0,
    "right_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": 2,
    "right_shoulder_roll_joint": 3,
    "left_shoulder_yaw_joint": 4,
    "right_shoulder_yaw_joint": 5,
    "left_elbow_pitch_joint": 6,
    "right_elbow_pitch_joint": 7,
    "left_wrist_yaw_joint": 8,
    "right_wrist_yaw_joint": 9,
    "left_wrist_roll_joint": 10,
    "right_wrist_roll_joint": 11,
    "left_wrist_pitch_joint": 12,
    "right_wrist_pitch_joint": 13,
    # hand joints
    "L_index_proximal_joint": 14,
    "L_middle_proximal_joint": 15,
    "L_pinky_proximal_joint": 16,
    "L_ring_proximal_joint": 17,
    "L_thumb_proximal_yaw_joint": 18,
    "R_index_proximal_joint": 19,
    "R_middle_proximal_joint": 20,
    "R_pinky_proximal_joint": 21,
    "R_ring_proximal_joint": 22,
    "R_thumb_proximal_yaw_joint": 23,
    "L_index_intermediate_joint": 24,
    "L_middle_intermediate_joint": 25,
    "L_pinky_intermediate_joint": 26,
    "L_ring_intermediate_joint": 27,
    "L_thumb_proximal_pitch_joint": 28,
    "R_index_intermediate_joint": 29,
    "R_middle_intermediate_joint": 30,
    "R_pinky_intermediate_joint": 31,
    "R_ring_intermediate_joint": 32,
    "R_thumb_proximal_pitch_joint": 33,
    "L_thumb_distal_joint": 34,
    "R_thumb_distal_joint": 35,
}
joint_names = list(joint_names_dict.keys())
tuned_joint_names = ["left-arm", "right-arm"]


@configclass
class ExhaustPipeGR1T2ClosedLoopEnvCfg(ExhaustPipeGR1T2BaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # replace the stiffness and dynamics in arm joints in the robot
        for joint_name in tuned_joint_names:
            self.scene.robot.actuators[joint_name].stiffness = 3000
            self.scene.robot.actuators[joint_name].damping = 100

        self.scene.robot_pov_cam = TiledCameraCfg(
            height=160,
            width=256,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.12, 1.85418), rot=(-0.17246, 0.98502, 0.0, 0.0), convention="ros"
            ),
            prim_path="{ENV_REGEX_NS}/RobotPOVCam",
            update_period=0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
        )

        self.actions.gr1_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=joint_names, scale=1.0, use_default_offset=False
        )
        self.viewer.eye = (0.0, 1.8, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 100
        self.decimation = 5
        self.sim.render_interval = 2
