# Downloading Datasets

The finetuning datasets are generated with Synethic Manipulation Motion Generation (SMMG), utilizing tools including
GR00T-Teleop, Mimic on Isaac Lab simulation environment. More details related to how datasets are generated could be viewed in [Isaac Lab Teleop & IL](https://isaac-sim.github.io/IsaacLab/main/source/overview/teleop_imitation.html).

## Available Datasets

Datasets are hosted on Hugging Face as listed below.

[nvidia/PhysicalAI-GR00T-Tuned-Tasks: Nut Pouring](https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks/tree/main/Nut-Pouring-task)

[nvidia/PhysicalAI-GR00T-Tuned-Tasks: Exhaust-Pipe-Sorting](https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks/tree/main/Exhaust-Pipe-Sorting-task)

You can download the GR00T-Lerobot format dataset ready for post training, or the original Mimic-generated HDF5 for data conversion.

## Download Instructions

Make sure you have registered your Hugging Face account and have read-access token ready.

```bash
# Provide your access token with read permission
huggingface-cli login

export DATASET="nvidia/PhysicalAI-GR00T-Tuned-Tasks"
# Define the path to save the datasets as DATASET_ROOT_DIR
huggingface-cli download --repo-type dataset --resume-download $DATASET  --local-dir $DATASET_ROOT_DIR

```

## Dataset Structure

`DATASET_ROOT_DIR` is the path to the directory where you want to store those assets as below.

<pre>
<code>
📂 PhysicalAI-GR00T-Tuned-Tasks
├── 📂 Exhaust-Pipe-Sorting-task
│   ├── 📂 data
│   ├── 📂 meta
│   └── 📂 videos
├── exhaust_pipe_sorting_task.hdf5
├── 📂 Nut-Pouring-task
│   ├── 📂 data
│   ├── 📂 meta
│   └── 📂 videos
├── nut_pouring_task.hdf5
└── README.md
</code>
</pre>
