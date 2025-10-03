# Installation

## Pre-requisites
- For [Policy Closed-loop Evaluation](#-policy-closed-loop-evaluation), we have tested on Ubuntu 22.04, GPU: L40, RTX 4090 and A6000 Ada, and Python==3.11, CUDA version 12.8.
- For [Policy Post Training](#post-training), see [GR00T-N1 pre-requisites](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#prerequisites)
- Please make sure you have the following dependencies installed in your system: `ffmpeg`, `libsm6`, `libxext6`

## Setup Development Environment
- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone --recurse-submodules git@github.com:isaac-sim/IsaacLabEvalTasks.git
```

- Using a python interpreter or conda/virtual env that has Isaac Lab installed, install the library required by [Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T/tree/n1-release)

```bash
# Within IsaacLabEvalTasks directory
cd submodules/Isaac-GR00T
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/IsaacLabEvalTasks/submodules/Isaac-GR00T
```

- Verify that the GR00T deps are correctly installed by running the following command:

```bash
python -c "import gr00t; print('gr00t imported successfully')"
```

- Using a python interpreter or conda/virtual env that has Isaac Lab installed, install the library of Evaluation Tasks

```bash
# Within IsaacLabEvalTasks directory
python -m pip install -e source/isaaclab_eval_tasks
```
