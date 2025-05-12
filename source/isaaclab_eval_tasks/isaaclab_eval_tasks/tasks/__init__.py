"""Package containing task implementations for various robotic environments."""

import gymnasium as gym
import os
import toml

from isaaclab_tasks.utils import import_packages

from .manipulation.pick_place import exhaustpipe_gr1t2_closedloop_env_cfg, nutpour_gr1t2_closedloop_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": exhaustpipe_gr1t2_closedloop_env_cfg.ExhaustPipeGR1T2ClosedLoopEnvCfg},
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
