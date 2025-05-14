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