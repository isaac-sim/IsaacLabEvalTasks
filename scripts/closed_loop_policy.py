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
