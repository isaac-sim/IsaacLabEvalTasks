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
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class EvalTaskConfig(Enum):
    NUTPOURING = (
        "Isaac-NutPour-GR1T2-ClosedLoop-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Nut-Pouring-task",
        (
            "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on"
            " the metallic measuring scale."
        ),
        "nut_pouring_task.hdf5",
        0   # 1 is reserved for data validity check, following GR00T-N1 guidelines.
    )
    PIPESORTING = (
        "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task",
        "Pick up the blue pipe and place it into the blue bin.",
        "exhaust_pipe_sorting_task.hdf5",
        2   # 1 is reserved for data validity check, following GR00T-N1 guidelines.
    )

    def __init__(self, task: str, model_path: str, language_instruction: str, hdf5_name: str, task_index: int):
        self.task = task
        self.model_path = model_path
        self.language_instruction = language_instruction
        self.hdf5_name = hdf5_name
        assert task_index != 1, "task_index must not be 1. (Use 0 for nutpouring, 2 for exhaustpipe, etc.)"
        self.task_index = task_index

@dataclass
class Gr00tN1ClosedLoopArguments:
    # Simulation specific parameters
    headless: bool = field(
        default=False, metadata={"description": "Whether to run the simulator in headless (no GUI) mode."}
    )
    num_envs: int = field(default=10, metadata={"description": "Number of environments to run in parallel."})
    enable_pinocchio: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to use Pinocchio for physics simulation. Required for NutPouring and ExhaustPipe tasks."
            )
        },
    )
    record_camera: bool = field(
        default=False,
        metadata={"description": "Whether to record the camera images as videos during evaluation."},
    )
    record_video_output_path: str = field(
        default="videos/",
        metadata={"description": "Path to save the recorded videos."},
    )

    # model specific parameters
    task_name: str = field(
        default="nutpouring", metadata={"description": "Short name of the task to run (e.g., nutpouring, exhaustpipe)."}
    )
    task: str = field(default="", metadata={"description": "Full task name for the gym-registered environment."})
    language_instruction: str = field(
        default="", metadata={"description": "Instruction given to the policy in natural language."}
    )
    model_path: str = field(default="", metadata={"description": "Full path to the tuned model checkpoint directory."})
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    embodiment_tag: str = field(
        default="gr1",
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference (e.g., 'gr1' or 'new_embodiment')."
            )
        },
    )
    denoising_steps: int = field(
        default=4, metadata={"description": "Number of denoising steps used in the policy inference."}
    )
    data_config: str = field(
        default="gr1_arms_only", metadata={"description": "Name of the data configuration to use for the policy."}
    )
    original_image_size: tuple[int, int, int] = field(
        default=(160, 256, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(256, 256, 3),
        metadata={"description": "Target size for images after resizing and padding as (height, width, channels)."},
    )
    gr00t_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr00t" / "gr00t_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )

    # robot (GR1) simulation specific parameters
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr1" / "action_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 action space in Lab."
            )
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr1" / "state_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 state space in Lab."
            )
        },
    )

    # Default to GPU policy and CPU physics simulation
    policy_device: str = field(
        default="cuda", metadata={"description": "Device to run the policy model on (e.g., 'cuda' or 'cpu')."}
    )
    simulation_device: str = field(
        default="cpu", metadata={"description": "Device to run the physics simulation on (e.g., 'cpu' or 'cuda')."}
    )

    # Evaluation parameters
    max_num_rollouts: int = field(
        default=100, metadata={"description": "Maximum number of rollouts to perform during evaluation."}
    )
    checkpoint_name: str = field(
        default="gr00t-n1-2b-tuned", metadata={"description": "Name of the model checkpoint used for evaluation."}
    )
    eval_file_path: Optional[str] = field(
        default=None, metadata={"description": "Path to the file where evaluation results will be saved."}
    )

    # Closed loop specific parameters
    num_feedback_actions: int = field(
        default=16,
        metadata={
            "description": "Number of feedback actions to execute per rollout (can be less than action_horizon)."
        },
    )
    rollout_length: int = field(default=30, metadata={"description": "Number of steps in each rollout episode."})
    seed: int = field(default=10, metadata={"description": "Random seed for reproducibility."})

    def __post_init__(self):
        # Populate fields from enum based on task_name
        if self.task_name.upper() not in EvalTaskConfig.__members__:
            raise ValueError(f"task_name must be one of: {', '.join(EvalTaskConfig.__members__.keys())}")
        config = EvalTaskConfig[self.task_name.upper()]
        if self.task == "":
            self.task = config.task
        if self.model_path == "":
            self.model_path = config.model_path
        if self.language_instruction == "":
            self.language_instruction = config.language_instruction
        # If model path is relative, return error
        if not os.path.isabs(self.model_path):
            raise ValueError("model_path must be an absolute path. Do not use relative paths.")
        assert (
            self.num_feedback_actions <= self.action_horizon
        ), "num_feedback_actions must be less than or equal to action_horizon"
        # assert all paths exist
        assert Path(self.gr00t_joints_config_path).exists(), "gr00t_joints_config_path does not exist"
        assert Path(self.action_joints_config_path).exists(), "action_joints_config_path does not exist"
        assert Path(self.state_joints_config_path).exists(), "state_joints_config_path does not exist"
        assert Path(self.model_path).exists(), "model_path does not exist."
        # embodiment_tag
        assert self.embodiment_tag in [
            "gr1",
            "new_embodiment",
        ], "embodiment_tag must be one of the following: " + ", ".join(["gr1", "new_embodiment"])


@dataclass
class Gr00tN1DatasetConfig:
    # Datasets & task specific parameters
    data_root: Path = field(
        default=Path("/mnt/datab/PhysicalAI-GR00T-Tuned-Tasks"),
        metadata={"description": "Root directory for all data storage."},
    )
    task_name: str = field(
        default="nutpouring", metadata={"description": "Short name of the task to run (e.g., nutpouring, exhaustpipe)."}
    )
    language_instruction: str = field(
        default="", metadata={"description": "Instruction given to the policy in natural language."}
    )
    hdf5_name: str = field(default="", metadata={"description": "Name of the HDF5 file to use for the dataset."})

    # Mimic-generated HDF5 datafield
    state_name_sim: str = field(
        default="robot_joint_pos", metadata={"description": "Name of the state in the HDF5 file."}
    )
    action_name_sim: str = field(
        default="processed_actions", metadata={"description": "Name of the action in the HDF5 file."}
    )
    pov_cam_name_sim: str = field(
        default="robot_pov_cam", metadata={"description": "Name of the POV camera in the HDF5 file."}
    )
    # Gr00t-LeRobot datafield
    state_name_lerobot: str = field(
        default="observation.state", metadata={"description": "Name of the state in the LeRobot file."}
    )
    action_name_lerobot: str = field(
        default="action", metadata={"description": "Name of the action in the LeRobot file."}
    )
    video_name_lerobot: str = field(
        default="observation.images.ego_view", metadata={"description": "Name of the video in the LeRobot file."}
    )
    task_description_lerobot: str = field(
        default="annotation.human.action.task_description",
        metadata={"description": "Name of the task description in the LeRobot file."},
    )
    valid_lerobot: str = field(
        default="annotation.human.action.valid", metadata={"description": "Name of the validity in the LeRobot file."}
    )

    # Parquet
    chunks_size: int = field(default=1000, metadata={"description": "Number of episodes per data chunk."})
    # mp4 video
    fps: int = field(default=20, metadata={"description": "Frames per second for video recording."})
    # Metadata files
    data_path: str = field(
        default="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        metadata={"description": "Template path for storing episode data files."},
    )
    video_path: str = field(
        default="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        metadata={"description": "Template path for storing episode video files."},
    )
    modality_template_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr00t" / "modality.json",
        metadata={"description": "Path to the modality template JSON file."},
    )
    modality_fname: str = field(
        default="modality.json", metadata={"description": "Filename for the modality JSON file."}
    )
    episodes_fname: str = field(
        default="episodes.jsonl", metadata={"description": "Filename for the episodes JSONL file."}
    )
    tasks_fname: str = field(default="tasks.jsonl", metadata={"description": "Filename for the tasks JSONL file."})
    info_template_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr00t" / "info.json",
        metadata={"description": "Path to the info template JSON file."},
    )
    info_fname: str = field(default="info.json", metadata={"description": "Filename for the info JSON file."})
    # GR00T policy specific parameters
    gr00t_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr00t" / "gr00t_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )
    robot_type: str = field(
        default="gr1_arms_only", metadata={"description": "Type of robot embodiment used in the policy fine-tuning."}
    )
    # robot (GR1) simulation specific parameters
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr1" / "action_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 action space in Lab."
            )
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "gr1" / "state_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 state space in Lab."
            )
        },
    )
    original_image_size: tuple[int, int, int] = field(
        default=(160, 256, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(256, 256, 3), metadata={"description": "Target size for images after resizing and padding."}
    )

    hdf5_file_path: Path = field(init=False)
    lerobot_data_dir: Path = field(init=False)
    task_index: int = field(init=False)     # task index for the task description in LeRobot file

    def __post_init__(self):

        # Populate fields from enum based on task_name
        if self.task_name.upper() not in EvalTaskConfig.__members__:
            raise ValueError(f"task_name must be one of: {', '.join(EvalTaskConfig.__members__.keys())}")
        config = EvalTaskConfig[self.task_name.upper()]
        self.language_instruction = config.language_instruction
        self.hdf5_name = config.hdf5_name
        self.task_index = config.task_index

        self.hdf5_file_path = self.data_root / self.hdf5_name
        self.lerobot_data_dir = self.data_root / self.hdf5_name.replace(".hdf5", "") / "lerobot"

        # Assert all paths exist
        assert self.hdf5_file_path.exists(), "hdf5_file_path does not exist"
        assert Path(self.gr00t_joints_config_path).exists(), "gr00t_joints_config_path does not exist"
        assert Path(self.action_joints_config_path).exists(), "action_joints_config_path does not exist"
        assert Path(self.state_joints_config_path).exists(), "state_joints_config_path does not exist"
        assert Path(self.info_template_path).exists(), "info_template_path does not exist"
        assert Path(self.modality_template_path).exists(), "modality_template_path does not exist"
        # if lerobot_data_dir not empty, throw a warning and remove
        if self.lerobot_data_dir.exists():
            print(f"Warning: lerobot_data_dir {self.lerobot_data_dir} already exists. Removing it.")
            # remove directory contents and the directory itself using shutil
            shutil.rmtree(self.lerobot_data_dir)
        # Prepare data keys for mimic-generated hdf5 file
        self.hdf5_keys = {
            "state": self.state_name_sim,
            "action": self.action_name_sim,
        }
        # Prepare data keys for LeRobot file
        self.lerobot_keys = {
            "state": self.state_name_lerobot,
            "action": self.action_name_lerobot,
            "video": self.video_name_lerobot,
            "annotation": (
                self.task_description_lerobot,
                self.valid_lerobot,
            ),
        }
