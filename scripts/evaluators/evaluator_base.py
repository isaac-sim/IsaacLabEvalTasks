# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
from typing import Optional, Dict

import gymnasium as gym


JSON_INDENT = 4


class EvaluatorBase(ABC):
    """
    Base class for all evaluators. An evaluator tracks the performance of a task over a series of demos.
    """

    def __init__(self,
                 checkpoint_name: str,
                 eval_file_path: Optional[str] = None,
                 seed: int = 10) -> None:
        """
        Initializes the EvaluatorBase object.

        Args:
            eval_file_path (os.path, optional): The path where the the evaluation file should be stored.
                                                Defaults to None (which means no evaluation file will be stored).
            checkpoint_name (str, optional): Name of checkpoint used for evaluation.
        """
        if eval_file_path is not None:
            assert os.path.exists(os.path.dirname(eval_file_path))
        self.eval_file_path = eval_file_path
        self.eval_dict = {}
        self.eval_dict["metadata"] = {
            "checkpoint_name": checkpoint_name,
            "seed": seed,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    @abstractmethod
    def evaluate_step(self, env: gym.Env) -> None:
        """
        Evaluates the current state of the task.

        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the cube stacking task is being evaluated.
        """
        pass

    def maybe_write_eval_file(self):
        """
        If the evaluation file is set, the eval dict will be written to it.
        """
        if self.eval_file_path is not None:
            with open(self.eval_file_path, 'w') as json_file:
                json.dump(self.eval_dict, json_file, indent=JSON_INDENT)

    @abstractmethod
    def summarize_demos(self) -> Dict:
        pass
