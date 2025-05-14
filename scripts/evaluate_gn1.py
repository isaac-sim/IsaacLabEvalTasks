# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from typing import Optional

import torch
import tqdm
import tyro

from isaaclab.app import AppLauncher
from isaacsim import SimulationApp


from config.args import Gr00tN1ClosedLoopArguments
args = tyro.cli(Gr00tN1ClosedLoopArguments)

if args.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401

# Launch the simulator
app_launcher = AppLauncher(headless=args.headless, enable_cameras=True, num_envs=args.num_envs, device=args.simulation_device)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_eval_tasks.tasks  # noqa: F401

from closed_loop_policy import create_sim_environment
from evaluators.gr00t_n1_evaluator import Gr00tN1Evaluator
from policies.gr00t_n1_policy import Gr00tN1Policy
from robot_joints import JointsAbsPosition


def run_closed_loop_policy(args: Gr00tN1ClosedLoopArguments,
                           simulation_app: SimulationApp,
                           env_cfg: ManagerBasedRLEnvCfg,
                           policy: Gr00tN1Policy,
                           evaluator: Optional[Gr00tN1Evaluator] = None):
    # Extract success checking function
    succeess_term = env_cfg.terminations.success
    # Disable terminations to avoid reset env
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(args.task, cfg=env_cfg)
    # Set seed
    env.unwrapped.seed(args.seed)

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            # if args.record_images or args.record_videos:
            #     record_camera = env.unwrapped.scene['record_cam']
            #     record_idx = 0
            # else:
            #     record_camera = None
            #     record_idx = None
            # Read the initial state of the world for this episode.
            # initial_state = episode_data["initial_state"]
            # env.unwrapped.reset_to(initial_state, None, is_relative=True)

            # Terminate the simulation_app if having enough rollouts counted by the evaluator
            # Otherwise, continue the rollout endlessly
            if evaluator is not None and evaluator.num_rollouts >= args.max_num_rollouts:
                break

            # reset environment
            env.unwrapped.sim.reset()
            env.reset(seed=args.seed)

            robot = env.unwrapped.scene['robot']
            robot_state_sim = JointsAbsPosition(robot.data.joint_pos,
                                                policy.gr1_state_joints_config,
                                                args.simulation_device)

            ego_camera = env.unwrapped.scene['robot_pov_cam']

            for _ in tqdm.tqdm(range(args.rollout_length)):
                robot_state_sim.set_joints_pos(robot.data.joint_pos)

                robot_action_sim = policy.get_new_goal(robot_state_sim, ego_camera, args.language_instruction)
                rollout_action = robot_action_sim.get_joints_pos(args.simulation_device)

                # Number of joints from policy shall match the env action reqs
                assert rollout_action.shape[-1] == env.action_space.shape[1]

                # take only the first num_feedback_actions, the rest are ignored, preventing over memorization
                for i in range(args.num_feedback_actions):
                    assert rollout_action[:, i, :].shape[0] == args.num_envs
                    env.step(rollout_action[:, i, :])

            # Check if rollout was successful
            if evaluator is not None:
                evaluator.evaluate_step(env, succeess_term)
                evaluator.summarize_demos()

    # Log evaluation results to a file
    if evaluator is not None:
        evaluator.maybe_write_eval_file()
    env.close()


if __name__ == "__main__":
    print("args", args)

    # model and environment related params
    gr00t_n1_policy = Gr00tN1Policy(args)
    env_cfg = create_sim_environment(args)
    evaluator = Gr00tN1Evaluator(args.checkpoint_name, args.eval_file_path, args.seed)

    # Run the closed loop policy.
    run_closed_loop_policy(args=args,
                           simulation_app=simulation_app,
                           env_cfg=env_cfg,
                           policy=gr00t_n1_policy,
                           evaluator=evaluator)

    # Close simulation app after rollout is complete
    simulation_app.close()
