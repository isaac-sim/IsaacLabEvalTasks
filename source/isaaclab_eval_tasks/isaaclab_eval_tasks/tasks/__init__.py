"""Package containing task implementations for various robotic environments."""

import os
import toml
import gymnasium as gym

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##

from .manipulation.pick_place import (
    nutpour_gr1t2_closedloop_env_cfg,
    exhaustpipe_gr1t2_closedloop_env_cfg,
)

gym.register(
    id="Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": exhaustpipe_gr1t2_closedloop_env_cfg.ExhaustPipeGR1T2ClosedLoopEnvCfg
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-NutPour-GR1T2-ClosedLoop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": nutpour_gr1t2_closedloop_env_cfg.NutPourGR1T2ClosedLoopEnvCfg,
    },
    disable_env_checker=True,
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
