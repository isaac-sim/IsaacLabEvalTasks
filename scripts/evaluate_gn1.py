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
from isaacsim import SimulationApp  # noqa: F401 # isort: skip
from isaaclab.app import AppLauncher  # noqa: F401 # isort: skip

import contextlib
import os
import torch
import tqdm
from typing import Optional

import tyro
from io_utils import VideoWriter

from config.args import Gr00tN1ClosedLoopArguments

args = tyro.cli(Gr00tN1ClosedLoopArguments)

if args.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401

# Launch the simulator
app_launcher = AppLauncher(
    headless=args.headless, enable_cameras=True, num_envs=args.num_envs, device=args.simulation_device
)
simulation_app = app_launcher.app

import gymnasium as gym

from closed_loop_policy import create_sim_environment
from evaluators.gr00t_n1_evaluator import Gr00tN1Evaluator
from policies.gr00t_n1_policy import Gr00tN1Policy
from policies.gt_policy import GTPolicy
from robot_joints import JointsAbsPosition

from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_eval_tasks.tasks  # noqa: F401


def run_closed_loop_policy(
    args: Gr00tN1ClosedLoopArguments,
    simulation_app: SimulationApp,
    env_cfg: ManagerBasedRLEnvCfg,
    policy: Gr00tN1Policy,
    evaluator: Optional[Gr00tN1Evaluator] = None,
):
    # Extract success checking function
    succeess_term = env_cfg.terminations.success
    # Disable terminations to avoid reset env
    env_cfg.terminations = {}

    # Create environment from loaded config
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    # Set seed
    env.seed(args.seed)

    # Create camera video recorder
    video_writer = None
    # Only record the first environment if multiple envs are running in one episode
    if args.record_camera:
        video_count = 0
        video_fpath = os.path.join(
            args.record_video_output_path, f"{args.task_name}_{args.checkpoint_name}_{video_count}.mp4"
        )
        video_writer = VideoWriter(
            video_fpath,
            # Height, width to be in the order of (width, height) for cv2
            args.original_image_size[:2][::-1],
            fps=20,
        )

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            # Terminate the simulation_app if having enough rollouts counted by the evaluator
            # Otherwise, continue the rollout endlessly
            if evaluator is not None and evaluator.num_rollouts >= args.max_num_rollouts:
                break

            # Reset environment
            env.sim.reset()
            env.reset()

            robot = env.scene["robot"]
            robot_state_sim = JointsAbsPosition(
                robot.data.joint_pos, policy.gr1_state_joints_config, args.simulation_device
            )

            ego_camera = env.scene["robot_pov_cam_groot"]

            if args.record_camera and video_writer is not None and os.path.exists(video_fpath):
                # Replace the last part of the video file name with the video count
                video_fpath = "_".join(video_fpath.split("_")[:-1]) + f"_{video_count}.mp4"
                video_writer.change_file_path(video_fpath)

            policy.reset()
            for _ in tqdm.tqdm(range(args.rollout_length)):
                robot_state_sim.set_joints_pos(robot.data.joint_pos)

                try:
                    robot_action_sim = policy.get_new_goal(robot_state_sim, ego_camera, args.language_instruction)
                except StopIteration:
                    break
                rollout_action = robot_action_sim.get_joints_pos(args.simulation_device)

                # Number of joints from policy shall match the env action reqs
                assert rollout_action.shape[-1] == env.action_space.shape[1]

                # take only the first num_feedback_actions, the rest are ignored, preventing over memorization
                for i in range(args.num_feedback_actions):
                    assert rollout_action[:, i, :].shape[0] == args.num_envs

                    env.step(rollout_action[:, i, :])

                    if args.record_camera and video_writer is not None:
                        # Only record the first environment if multiple envs are running
                        video_writer.add_image(ego_camera.data.output["rgb"][0])

            if args.record_camera and video_writer is not None:
                video_count += 1

            # Check if rollout was successful
            if evaluator is not None:
                evaluator.evaluate_step(env, succeess_term)
                evaluator.summarize_demos()

    # Log evaluation results to a file
    if evaluator is not None:
        evaluator.maybe_write_eval_file()
    if video_writer is not None:
        video_writer.close()
    env.close()


if __name__ == "__main__":
    print("args", args)

    # model and environment related params
    if args.model_path == "":
        assert args.dataset_path != ""
        print("No model path, but dataset path, using GTPolicy")
        gr00t_n1_policy = GTPolicy(args)
    else:
        print("model path, using Gr00tN1Policy")
        gr00t_n1_policy = Gr00tN1Policy(args)
    env_cfg = create_sim_environment(args)
    evaluator = Gr00tN1Evaluator(args.checkpoint_name, args.eval_file_path, args.seed)

    # Run the closed loop policy.
    run_closed_loop_policy(
        args=args, simulation_app=simulation_app, env_cfg=env_cfg, policy=gr00t_n1_policy, evaluator=evaluator
    )

    # Close simulation app after rollout is complete
    simulation_app.close()
