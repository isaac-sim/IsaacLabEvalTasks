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

"""Package containing task implementations for various robotic environments."""

import gymnasium as gym

from isaaclab_tasks.utils import import_packages

from .manipulation.pick_place import exhaustpipe_gr1t2_closedloop_env_cfg, nutpour_gr1t2_closedloop_env_cfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": exhaustpipe_gr1t2_closedloop_env_cfg.ExhaustPipeGR1T2ClosedLoopEnvCfg},
)

gym.register(
    id="Isaac-NutPour-GR1T2-ClosedLoop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": nutpour_gr1t2_closedloop_env_cfg.NutPourGR1T2ClosedLoopEnvCfg,
    },
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
