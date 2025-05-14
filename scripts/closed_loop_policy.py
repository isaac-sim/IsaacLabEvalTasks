# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import os
import random

import torch
import numpy as np

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from config.args import Gr00tN1ClosedLoopArguments


def create_sim_environment(args: Gr00tN1ClosedLoopArguments):
    """
    Creates a simulation environment based on the given arguments.

    Args:
        args (Gr00tN1ClosedLoopArguments): The arguments for the simulation environment.

    Returns:
        gym.Env: The created simulation environment.
    """
    env_name = args.task
    env_cfg = parse_env_cfg(env_name, device=args.simulation_device, num_envs=args.num_envs)

    # Improve appearance of environment
    # Add base environment
    # if self.args.background_env_usd_path is not None:
    #     env_cfg.scene.base_env = AssetBaseCfg(
    #         prim_path="/World/Background",
    #         init_state=AssetBaseCfg.InitialStateCfg(
    #             pos=[0, 0, -1.05
    #                 ]),    # -1.05 is the 0 height of the table, maybe make it a user argument.
    #         spawn=UsdFileCfg(usd_path=self.args.background_env_usd_path),
    #     )

    # Add recording camera
    camera_params = {
        "focal_length": 18.0,
        "position": (1.0, -0.25, 0.9)
    }
    if args.record_images or args.record_videos:
        env_cfg.scene.record_cam = None
        # env_cfg.scene.record_cam = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/record_cam",
        #     update_period=0.0333,
        #     height=1200,
        #     width=1200,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(focal_length=camera_params["focal_length"],
        #                                      focus_distance=400.0,
        #                                      horizontal_aperture=20.955,
        #                                      clipping_range=(0.1, 1.0e5)),
        #     offset=CameraCfg.OffsetCfg(pos=camera_params["position"],
        #                                rot=(0.755, 0.354, 0.228, 0.502),
        #                                convention="opengl"),
        # )

        if args.record_camera_output_path is not None:
            # Ensure directory exists
            os.makedirs(args.record_camera_output_path, exist_ok=True)

    # Disable all recorders
    env_cfg.recorders = {}

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return env_cfg
