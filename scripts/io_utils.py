# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import json
import collections
from pathlib import Path
from typing import List, Dict, Any
import yaml

import numpy as np


def dump_jsonl(data: collections.abc.Sequence, file_path: str | Path):
    assert isinstance(data, collections.abc.Sequence) and not isinstance(data, str)
    if isinstance(data, (np.ndarray, np.number)):
        data = data.tolist()
    with open(file_path, "w") as fp:
        for line in data:
            print(json.dumps(line), file=fp, flush=True)


def dump_json(data: np.ndarray | np.number | List, file_path: str | Path, **kwargs):
    if isinstance(data, (np.ndarray, np.number)):
        data = data.tolist()
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)


def load_gr1_joints_config(yaml_path: str | Path) -> Dict[str, Any]:
    """Load GR1 joint configuration from YAML file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('joints', {})