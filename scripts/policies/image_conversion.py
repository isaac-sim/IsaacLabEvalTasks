# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import cv2
import numpy as np
import torch


def resize_frames_with_padding(frames: torch.Tensor,
                               target_image_size: tuple,
                               bgr_conversion: bool = False,
                               pad_img: bool = True) -> torch.Tensor:
    """Process batch of frames with padding and resizing vectorized
    Args:
        frames: np.ndarray of shape [N, 256, 160, 3]
        target_image_size: target size (height, width)
        bgr_conversion: whether to convert BGR to RGB
        pad_img: whether to resize images
    """
    # create copy to cpu numpy array
    frames = frames.cpu().numpy()

    if bgr_conversion:
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    if pad_img:
        top_padding = (frames.shape[2] - frames.shape[1]) // 2
        bottom_padding = top_padding

        # Add padding to all frames at once
        frames = np.pad(
            frames,
            pad_width=((0, 0), (top_padding, bottom_padding), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )

    # Resize all frames at once
    if frames.shape[1:] != target_image_size:
        frames = np.stack([cv2.resize(f, target_image_size) for f in frames])

    return frames