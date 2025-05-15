# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import numpy as np
import random
import torch

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

    # Disable all recorders
    env_cfg.recorders = {}

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return env_cfg
