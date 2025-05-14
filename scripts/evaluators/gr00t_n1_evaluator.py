# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Optional

import torch
import gymnasium as gym

from isaaclab.managers import TerminationTermCfg as DoneTerm

from evaluators.evaluator_base import EvaluatorBase


class Gr00tN1Evaluator(EvaluatorBase):
    """
    The purpose of this class is to evaluate the performance of gr00t-N1 policy on the nut pouring
    and pipe sorting tasks, by tracking the success rate of the policy over a series of demos.
    Success is defined as termination term in the environment configuration script
    """
    def __init__(self, checkpoint_name: str, eval_file_path: Optional[str] = None, seed: int = 10) -> None:
        super().__init__(checkpoint_name, eval_file_path, seed)
        self.num_success = 0
        self.num_rollouts = 0

    def evaluate_step(self, env: gym.Env, succeess_term: DoneTerm) -> None:
        success_term_val = succeess_term.func(env, **succeess_term.params)

        self.num_success += torch.sum(success_term_val).item()
        self.num_rollouts += len(success_term_val)

    def summarize_demos(self):
        # printe in terminal with a table layout
        print(f"\n{'='*50}")
        print(f"\nSuccessful trials: {self.num_success}, out of {self.num_rollouts} trials")
        print(f"Success rate: {self.num_success / self.num_rollouts}")
        print(f"{'='*50}\n")

        self.eval_dict["summary"] = {
            "successful_trials": self.num_success,
            "total_rollouts": self.num_rollouts,
            "success_rate": self.num_success / self.num_rollouts
        }
