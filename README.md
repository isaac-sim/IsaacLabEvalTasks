# Isaac Lab Evaluation Tasks

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-green.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

##  📝 Overview

This repository introduces two new industrial manipulation tasks designed in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html), enabling simulating and evaluating manipulation policies (e.g. [Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T/tree/n1-release)) using a humanoid robot. The tasks are designed to simulate realistic industrial scenarios, including Nut Pouring and Exhaust Pipe Sorting.
It also provides benchmarking scripts for closed-loop evaluation of manipulation policy (i.e. Isaac GR00T N1) with post-trained checkpoints. These scripts enable developers to load prebuilt Isaac Lab environments and industrial tasks—such as nut pouring and pipe sorting—and run standardized benchmarks to quantitatively assess policy performance.

## 📦 Installation

For detailed installation instructions, see [Installation Guide](doc/installation.md).

## 🛠️ Evaluation Tasks

Two industrial tasks have been created in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) to simulate robotic manipulation scenarios. The environments are set up with a humanoid robot (i.e. Fourier GR1-T2) positioned in front of several industrial objects on a table. This can include multi-step bi-manual tasks such as grasping, moving, sorting, or placing the objects into specific locations.

The robot is positioned upright, facing the table with both arms slightly bent and hands open. A first-person-view monocular RGB camera is mounted on its head to cover the workspace.


### Nut Pouring

<div align="center">
<img src="doc/gr-1_nut_pouring_policy.gif" width="600" alt="Nut Pouring Task">
<p><em>The robot picks up a beaker containing metallic nuts, pours one nut into a bowl, and places the bowl on a scale.</em></p>
</div>

The task is defined as successful if following criteria have been met.
1. The sorting beaker is placed in the sorting bin
2. The factory nut is in the sorting bowl
3. The sorting bowl is placed on the sorting scale


### Exhaust Pipe Sorting

<div align="center">
<img src="doc/gr-1_exhaust_pipe_demo.gif" width="600" alt="Exhaust Pipe Sorting Task">
<p><em>The robot picks up the blue exhaust pipe, transfers it to the other hand, and places the pipe into the blue bin.</em></p>
</div>

The task is defined as successful if following criteria has been met.

1. The blue exhaust pipe is placed in the correct position


## 📦 Downloading Datasets (Optional)

For dataset information and download instructions, see [Datasets Guide](doc/datasets.md).

## 🤖 Isaac GR00T N1 Policy Post Training (Optional)

For detailed post-training instructions including data conversion and model fine-tuning, see [Post Training Guide](doc/post-training.md).


## 📦 Downloading Checkpoints

For information on available pre-trained checkpoints and download instructions, see [Checkpoints Guide](doc/checkpoints.md).

## 📈 Policy Closed-loop Evaluation

For detailed evaluation instructions including benchmarking features and performance results, see [Evaluation Guide](doc/evaluation.md).

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

For common issues and solutions, see [Troubleshooting Guide](doc/troubleshooting.md).

## Contributing
For more details, see [CONTRIBUTING.md](CONTIRBUTING.md)
