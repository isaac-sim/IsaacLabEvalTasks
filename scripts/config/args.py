# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path


class EvalTaskConfig(Enum):
    NUTPOURING = (
        "Isaac-NutPour-GR1T2-ClosedLoop-v0",
        "/mnt/datab/gr00t-mimic/ckpts/nut_nbke_vpd_v34_1k_5demo_gr1_3e-3_bs96_20ksteps_8gpus_h100-1/master/checkpoint-20000",
        "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on the metallic measuring scale."
    )
    PIPE_SORTING = (
        "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
        "/mnt/datab/gr00t-mimic/ckpts/pipe_nobke_vpd_v10_1k_5dem_gr1_3e-3_bs96_20ksteps_8gpus_h100-2/master/checkpoint-20000",
        "Pickup the blue pipe and place it into the blue bin."
    )

    def __init__(self, task, model_path, language_instruction):
        self.task = task
        self.model_path = model_path
        self.language_instruction = language_instruction


@dataclass
class Gr00tN1ClosedLoopArguments():
    record_images: bool = True
    record_videos: bool = True
    num_envs: int = 2
    background_env_usd_path: Optional[str] = None
    record_camera_output_path: Optional[str] = None
    enable_pinocchio: bool = True

    # model specific parameters
    task_name: str = "nutpouring"
    task: str = ""
    language_instruction: str = ""
    model_path: str = ""
    embodiment_tag: str = "gr1"
    action_horizon: int = 16
    denoising_steps: int = 4

    data_config: str = "gr1_arms_only"
    original_image_size: tuple[int, int, int] = (160, 256, 3)
    target_image_size: tuple[int, int, int] = (256, 256, 3)
    gr00t_joints_config_path: Path = Path(__file__).parent.resolve() / "gr00t" / "gr00t_joint_space.yaml"

    # robot (GR1) simulation specific parameters
    action_joints_config_path: Path = Path(__file__).parent.resolve() / "gr1" / "action_joint_space.yaml"
    state_joints_config_path: Path = Path(__file__).parent.resolve() / "gr1" / "state_joint_space.yaml"

    # Default to GPU policy and CPU physics simulation
    policy_device: str = "cuda"
    simulation_device: str = "cpu"

    # Evaluation parameters
    max_num_rollouts: int = 2
    checkpoint_name: str = "gr00t-n1"
    eval_file_path: Optional[str] = f'/tmp/{checkpoint_name}-{task}.json'

    # Simulator specific parameters
    headless: bool = False
    # could be less than action_horizon
    num_feedback_actions: int = 16
    rollout_length: int = 30
    seed: int = 10

    def __post_init__(self):
        # Populate fields from enum based on task_name
        if self.task_name.upper() not in EvalTaskConfig.__members__:
            raise ValueError(
                f"task_name must be one of: {', '.join(EvalTaskConfig.__members__.keys())}"
            )
        config = EvalTaskConfig[self.task_name.upper()]
        self.task = config.task
        self.model_path = config.model_path
        self.language_instruction = config.language_instruction

        assert self.num_feedback_actions <= self.action_horizon, (
            "num_feedback_actions must be less than or equal to action_horizon"
        )
        # assert all paths exist
        assert Path(self.gr00t_joints_config_path).exists(), (
            "gr00t_joints_config_path does not exist"
        )
        assert Path(self.action_joints_config_path).exists(), (
            "action_joints_config_path does not exist"
        )
        assert Path(self.state_joints_config_path).exists(), (
            "state_joints_config_path does not exist"
        )
        assert Path(self.model_path).exists(), (
            "model_path does not exist"
        )
        # embodiment_tag
        assert self.embodiment_tag in ["gr1", "new_embodiment"], (
            "embodiment_tag must be one of the following: " + ", ".join(["gr1", "new_embodiment"])
        )
