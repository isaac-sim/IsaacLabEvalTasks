# Isaac Lab Evaluation Tasks

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository helps evaluating manipulation policies (e.g. Gr00t-N1) trained/post-trained in Issac Lab, over pre-defined tasks. We
have provided 2 industrial humanoid manipulation tasks, Nut Pouring and Exhaust Pipe Sorting.


## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone --recurse-submodules https://github.com/isaac-sim/IsaacLabEvalTasks.git

# Option 2: SSH
git clone --recurse-submodules git@github.com:isaac-sim/IsaacLabEvalTasks.git
```

- Using a python interpreter that has Isaac Lab installed, install the library required by [Isaac Gr00t](https://github.com/NVIDIA/Isaac-GR00T)

```bash
cd submodules/Isaac-GR00T
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
```

- Verify that the GR00t deps are correctly installed by running the following command:

```bash
python -c "import gr00t; print('gr00t imported successfully')"
```

## Downloading Checkpoint and Dataset
If you are 


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Hugging Face
