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
