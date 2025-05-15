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

import h5py
import json
import multiprocessing as mp
import numpy as np
import shutil
import subprocess
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict

import pandas as pd
import torchvision
import tyro
from io_utils import dump_json, dump_jsonl, load_gr1_joints_config, load_json
from policies.image_conversion import resize_frames_with_padding
from policies.joints_conversion import remap_sim_joints_to_policy_joints
from robot_joints import JointsAbsPosition

from config.args import Gr00tN1DatasetConfig


def get_video_metadata(video_path: str) -> Dict[str, Any] | None:
    """
    Get video metadata in the specified format.

    Args:
        video_path: Path to the video file.

    Returns:
        Video metadata including shape, names, and video_info, or None if an error occurs.
    """
    # Get video information using ffprobe
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]

        # Parse frame rate (comes as fraction like "15/1")
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        # Check for audio streams
        audio_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            video_path,
        ]
        audio_output = subprocess.check_output(audio_cmd).decode("utf-8")
        audio_data = json.loads(audio_output)
        has_audio = len(audio_data.get("streams", [])) > 0

        metadata = {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],  # Assuming 3 channels
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.is_depth_map": False,
                "has_audio": has_audio,
            },
        }

        return metadata

    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return None


def get_feature_info(step_data: pd.DataFrame, video_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Get feature info from each  frame of the video.

    Args:
        step_data: DataFrame containing data of an episode.
        video_paths: Dictionary mapping video keys to their file paths.

    Returns:
        Dictionary containing feature information for each column and video.
    """
    features = {}
    for video_key, video_path in video_paths.items():
        video_metadata = get_video_metadata(video_path)
        features[video_key] = video_metadata
    assert isinstance(step_data, pd.DataFrame)
    for column in step_data.columns:
        column_data = np.stack(step_data[column], axis=0)
        shape = column_data.shape
        if len(shape) == 1:
            shape = (1,)
        else:
            shape = shape[1:]
        features[column] = {
            "dtype": column_data.dtype.name,
            "shape": shape,
        }
        # State & action
        if column in [LEROBOT_KEY["state"], LEROBOT_KEY["action"]]:
            dof = column_data.shape[1]
            features[column]["names"] = [f"motor_{i}" for i in range(dof)]

    return features


def generate_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    config: Gr00tN1DatasetConfig,
    step_data: pd.DataFrame,
    video_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Generate the info.json data field.

    Args:
        total_episodes: Total number of episodes in the dataset.
        total_frames: Total number of frames across all episodes.
        total_tasks: Total number of tasks in the dataset.
        total_videos: Total number of videos in the dataset.
        total_chunks: Total number of data chunks.
        config: Configuration object containing dataset and path information.
        step_data: DataFrame containing step data for an example episode.
        video_paths: Dictionary mapping video keys to their file paths.

    Returns:
        Dictionary containing the info.json data field.
    """

    info_template = load_json(config.info_template_path)

    info_template["robot_type"] = config.robot_type
    info_template["total_episodes"] = total_episodes
    info_template["total_frames"] = total_frames
    info_template["total_tasks"] = total_tasks
    info_template["total_videos"] = total_videos
    info_template["total_chunks"] = total_chunks
    info_template["chunks_size"] = config.chunks_size
    info_template["fps"] = config.fps

    info_template["data_path"] = config.data_path
    info_template["video_path"] = config.video_path

    features = get_feature_info(step_data, video_paths)

    info_template["features"] = features
    return info_template


def write_video_job(queue: mp.Queue, error_queue: mp.Queue, config: Gr00tN1DatasetConfig) -> None:
    """
    Write frames to videos in mp4 format.

    Args:
        queue: Multiprocessing queue containing video frame data to be written.
        error_queue: Multiprocessing queue for reporting errors from worker processes.
        config: Configuration object containing dataset and path information.

    Returns:
        None
    """
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            video_path, frames, fps, video_type = job
            if video_type == "image":
                # Create parent directory if it doesn't exist
                video_path.parent.mkdir(parents=True, exist_ok=True)
                assert frames.shape[1:] == config.original_image_size, f"Frames shape is {frames.shape}"
                frames = resize_frames_with_padding(
                    frames, target_image_size=config.target_image_size, bgr_conversion=False, pad_img=True
                )
                # h264 codec encoding
                torchvision.io.write_video(video_path, frames, fps, video_codec="h264")

        except Exception as e:
            # Get the traceback
            error_queue.put(f"Error in process: {e}\n{traceback.format_exc()}")


def convert_trajectory_to_df(
    trajectory: h5py.Dataset,
    episode_index: int,
    index_start: int,
    config: Gr00tN1DatasetConfig,
) -> Dict[str, Any]:
    """
    Convert a single trajectory from HDF5 to a pandas DataFrame.

    Args:
        trajectory: HDF5 dataset containing trajectory data.
        episode_index: Index of the current episode.
        index_start: Starting index for the episode.
        config: Configuration object containing dataset and path information.

    Returns:
        Dictionary containing the DataFrame, annotation set, and episode length.
    """

    return_dict = {}
    data = {}

    gr00t_modality_config = load_json(config.modality_template_path)

    gr00t_joints_config = load_gr1_joints_config(config.gr00t_joints_config_path)
    action_joints_config = load_gr1_joints_config(config.action_joints_config_path)
    state_joints_config = load_gr1_joints_config(config.state_joints_config_path)

    # 1. Get state, action, and timestamp
    length = None
    for key, hdf5_key_name in config.hdf5_keys.items():
        assert key in ["state", "action"]
        lerobot_key_name = config.lerobot_keys[key]
        if key == "state":
            joints = trajectory["obs"][hdf5_key_name]
        else:
            joints = trajectory[hdf5_key_name]
        joints = joints.astype(np.float64)

        if key == "state":
            # remove the last obs
            joints = joints[:-1]
            input_joints_config = state_joints_config
        elif key == "action":
            # remove the last idle action
            joints = joints[:-1]
            input_joints_config = action_joints_config
        else:
            raise ValueError(f"Unknown key: {key}")
        assert joints.ndim == 2
        assert joints.shape[1] == len(input_joints_config)

        # 1.1. Remap the joints to the LeRobot joint orders
        joints = JointsAbsPosition.from_array(joints, input_joints_config, device="cpu")
        remapped_joints = remap_sim_joints_to_policy_joints(joints, gr00t_joints_config)
        # 1.2. Fill in the missing joints with zeros
        ordered_joints = []
        for joint_group in gr00t_modality_config[key].keys():
            num_joints = (
                gr00t_modality_config[key][joint_group]["end"] - gr00t_modality_config[key][joint_group]["start"]
            )
            if joint_group not in remapped_joints.keys():
                remapped_joints[joint_group] = np.zeros(
                    (joints.get_joints_pos().shape[0], num_joints), dtype=np.float64
                )
            else:
                assert remapped_joints[joint_group].shape[1] == num_joints
            ordered_joints.append(remapped_joints[joint_group])

        # 1.3. Concatenate the arrays for parquets
        concatenated = np.concatenate(ordered_joints, axis=1)
        data[lerobot_key_name] = [row for row in concatenated]

    assert len(data[config.lerobot_keys["action"]]) == len(data[config.lerobot_keys["state"]])
    length = len(data[config.lerobot_keys["action"]])
    data["timestamp"] = np.arange(length).astype(np.float64) * (1.0 / config.fps)
    # 2. Get the annotation
    annotation_keys = config.lerobot_keys["annotation"]
    # task selection
    data[annotation_keys[0]] = np.ones(length, dtype=int) * config.task_index
    # valid is 1
    data[annotation_keys[1]] = np.ones(length, dtype=int) * 1

    # 3. Other data
    data["episode_index"] = np.ones(length, dtype=int) * episode_index
    data["task_index"] = np.zeros(length, dtype=int)
    data["index"] = np.arange(length, dtype=int) + index_start
    # last frame is successful
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = 1
    done = np.zeros(length, dtype=bool)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done

    dataframe = pd.DataFrame(data)

    return_dict["data"] = dataframe
    return_dict["length"] = length
    return_dict["annotation"] = set(data[annotation_keys[0]]) | set(data[annotation_keys[1]])
    return return_dict


def convert_hdf5_to_lerobot(config: Gr00tN1DatasetConfig):
    """
    Convert the MimcGen HDF5 dataset to Gr00t-LeRobot format.

    Args:
        config: Configuration object containing dataset and path information.

    Returns:
        None
    """
    # Create a queue to communicate with the worker processes
    max_queue_size = 10
    num_workers = 4
    queue = mp.Queue(maxsize=max_queue_size)
    error_queue = mp.Queue()  # for error handling
    # Start the worker processes
    workers = []
    for _ in range(num_workers):
        worker = mp.Process(target=write_video_job, args=(queue, error_queue, config))
        worker.start()
        workers.append(worker)

    assert Path(config.hdf5_file_path).exists()
    hdf5_handler = h5py.File(config.hdf5_file_path, "r")
    hdf5_data = hdf5_handler["data"]

    # 1. Generate meta/ folder
    config.lerobot_data_dir.mkdir(parents=True, exist_ok=True)
    lerobot_meta_dir = config.lerobot_data_dir / "meta"
    lerobot_meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = {1: "valid"}
    tasks.update({config.task_index: f"{config.language_instruction}"})

    # 2. Generate data/
    total_length = 0
    example_data = None
    video_paths = {}

    trajectory_ids = list(hdf5_data.keys())
    episodes_info = []
    for episode_index, trajectory_id in enumerate(tqdm(trajectory_ids)):

        try:
            trajectory = hdf5_data[trajectory_id]
            df_ret_dict = convert_trajectory_to_df(
                trajectory=trajectory, episode_index=episode_index, index_start=total_length, config=config
            )
        except Exception as e:
            print(f"Error loading trajectory {trajectory_id}: {e}")
            continue
        # 2.1. Save the episode data
        dataframe = df_ret_dict["data"]
        episode_chunk = episode_index // config.chunks_size
        save_relpath = config.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
        save_path = config.lerobot_data_dir / save_relpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(save_path)

        # 2.2. Update total length, episodes_info
        length = df_ret_dict["length"]
        total_length += length
        episodes_info.append(
            {
                "episode_index": episode_index,
                "tasks": [tasks[task_index] for task_index in df_ret_dict["annotation"]],
                "length": length,
            }
        )
        # 2.3. Generate videos/
        new_video_relpath = config.video_path.format(
            episode_chunk=episode_chunk, video_key=config.lerobot_keys["video"], episode_index=episode_index
        )
        new_video_path = config.lerobot_data_dir / new_video_relpath
        if config.video_name_lerobot not in video_paths.keys():
            video_paths[config.video_name_lerobot] = new_video_path
        assert config.pov_cam_name_sim in trajectory["obs"].keys()
        frames = np.array(trajectory["obs"][config.pov_cam_name_sim])
        # remove last frame due to how Lab reports observations
        frames = frames[:-1]
        assert len(frames) == length
        queue.put((new_video_path, frames, config.fps, "image"))

        if example_data is None:
            example_data = df_ret_dict

    # 3. Generate the rest of meta/
    # 3.1. Generate tasks.json
    tasks_path = lerobot_meta_dir / config.tasks_fname
    task_jsonlines = [{"task_index": task_index, "task": task} for task_index, task in tasks.items()]
    dump_jsonl(task_jsonlines, tasks_path)

    # 3.2. Generate episodes.jsonl
    episodes_path = lerobot_meta_dir / config.episodes_fname
    dump_jsonl(episodes_info, episodes_path)

    # 3.3. Generate modality.json
    modality_path = lerobot_meta_dir / config.modality_fname
    shutil.copy(config.modality_template_path, modality_path)

    # # 3.4. Generate info.json
    info_json = generate_info(
        total_episodes=len(trajectory_ids),
        total_frames=total_length,
        total_tasks=len(tasks),
        total_videos=len(trajectory_ids),
        total_chunks=len(trajectory_ids) // config.chunks_size,
        step_data=example_data["data"],
        video_paths=video_paths,
        config=config,
    )
    dump_json(info_json, lerobot_meta_dir / "info.json", indent=4)

    try:
        # Check for errors in the error queue
        while not error_queue.empty():
            error_message = error_queue.get()
            print(f"Error in worker process: {error_message}")

        # Stop the worker processes
        for _ in range(num_workers):
            queue.put(None)
        for worker in workers:
            worker.join()

        # Close the HDF5 file handler
        hdf5_handler.close()

    except Exception as e:
        print(f"Error in main process: {e}")
        # Make sure to clean up even if there's an error
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        if not hdf5_handler.closed:
            hdf5_handler.close()
        raise  # Re-raise the exception after cleanup


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(Gr00tN1DatasetConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T LEROBOT DATASET CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")
    convert_hdf5_to_lerobot(config)
