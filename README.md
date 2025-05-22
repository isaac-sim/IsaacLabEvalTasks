# Isaac Lab Evaluation Tasks

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

##  📝 Overview

This repository introduces two new industrial manipulation tasks designed in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html), enabling simulating and evaluating manipulation policies (e.g. [Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T)) using a humanoid robot. The tasks are designed to simulate realistic industrial scenarios, including Nut Pouring and Exhaust Pipe Sorting.
It also provides benchmarking scripts for closed-loop evaluation of manipulation policy (i.e. Isaac GR00T N1) with post-trained checkpoints. These scripts enable developers to load prebuilt Issac Lab environments and industrial tasks—such as nut pouring and pipe sorting—and run standardized benchmarks to quantitatively assess policy performance.

## 📦 Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone --recurse-submodules https://github.com/isaac-sim/IsaacLabEvalTasks.git

# Option 2: SSH
git clone --recurse-submodules git@github.com:isaac-sim/IsaacLabEvalTasks.git
```

- Using a python interpreter or conda/virtual env that has Isaac Lab installed, install the library required by [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)

```bash
cd submodules/Isaac-GR00T
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/IsaacLabEvalTasks/submodules/Isaac-GR00T
```

- Verify that the GR00t deps are correctly installed by running the following command:

```bash
python -c "import gr00t; print('gr00t imported successfully')"
```

- Using a python interpreter or conda/virtual env that has Isaac Lab installed, install the library of Evaluation Tasks

```bash
python -m pip install -e source/isaaclab_eval_tasks
```

## 🛠️ Evaluation Tasks

Two industrial tasks have been created in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) to simulate robotic manipulation scenarios. The environments are set up with a humanoid robot (i.e. Fourier GR1-T2) positioned in front of several industrial objects on a table. This can include multi-step bi-manual tasks such as grasping, moving, sorting, or placing the objects into specific locations.

The robot is positioned upright, facing the table with both arms slightly bent and hands open. A first-person-view monocular RGB camera is mounted on its head to cover the workspace.


### Nut Pouring

<div align="center">
<img src="doc/gr-1_nut_pouring_policy.gif" width="600" alt="Nut Pouring Task">
<p><em>The robot picks up a beaker containing metallic nuts, pours one nut into a bowl, and places the bowl on a scale.</em></p>
</div>

The task is defined as successful if following criterias have been met.
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

The finetuning datasets are generated with Synethic Manipulation Motion Generation (SMMG), utilizing tools including
GR00T-Teleop, Mimic on Isaac Lab simulation environment. More details related to how datasets are generated could be viewed in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).

Datasets are hosted on Hugging Face as listed below.

[nvidia/PhysicalAI-GR00T-Tuned-Tasks: Nut Pouring](https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks/tree/main/Nut-Pouring-task)

[nvidia/PhysicalAI-GR00T-Tuned-Tasks: Exhaust-Pipe-Sorting](https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks/tree/main/Exhaust-Pipe-Sorting-task)

You can download the GR00T-Lerobot format dataset ready for post training, or the original Mimic-generated HDF5 for data conversion.

Make sure you have registered your Hugging Face account and have read-access token ready.

```bash
# Provide your access token with read permission
huggingface-cli login

DATASET="nvidia/PhysicalAI-GR00T-Tuned-Tasks"
huggingface-cli download --repo-type dataset --resume-download $DATASET  --local-dir $DATASET_ROOT_DIR

```
`DATASET_ROOT_DIR` is the path to the directory where you want to store those assets as below.

<pre>
<code>
📂 PhysicalAI-GR00T-Tuned-Tasks
├── 📂 Exhaust-Pipe-Sorting-task
│   ├── 📂 data
│   ├── 📂 meta
│   └── 📂 videos
├── exhaust_pipe_sorting_task.hdf5
├── 📂 Nut-Pouring-task
│   ├── 📂 data
│   ├── 📂 meta
│   └── 📂 videos
├── nut_pouring_task.hdf5
└── README.md
</code>
</pre>

## 🤖 Isaac GR00T N1 Policy Post-Trainig (Optional)

[GR00T N1](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#nvidia-isaac-gr00t-n1) is a foundation model for generalized humanoid robot reasoning and skills, trained on an extensive multimodal dataset that includes real-world, synthetic, and internet-scale data. The model is designed for cross-embodiment generalization and can be efficiently adapted to new robot embodiments, tasks, and environments through post-training.

We followed the recommended GR00T N1 post-training workflow to adapt the model for the Fourier GR1 robot, targeting two industrial manipulation tasks: nut pouring and exhaust pipe sorting. The process involves multiple steps introduced below. You can also skip to the next section [Downloading Checkpoints](#downloading-checkpoints) to get post-trained checkpoints.

### Data Conversion

The process involved converting demonstration data (Mimic-generated motion trajectories in HDF5) into the LeRobot-compatible schema ([GR00T-Lerobot format guidlines](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md)).


- Using a python interpreter or conda/virtual env that has Isaac Lab, GR00T and Eavluation Tasks installed, convert Mimic-generated trajectories by

```bash
# Example: Set `task_index` Based on Task
# Nut Pouring
export TASK_INDEX=0
# Uncomment the below is Task is Exhaust Pipe Sorting
# export TASK_INDEX=2
# data_root is directory of where Mimic-generated HDF5 is saved locally
python scripts/convert_hdf5_to_lerobot.py --task_index $TASK_INDEX --data_root $DATASET_ROOT_DIR
```
The GR00T-LeRobot-compatible datasets will be available in `DATASET_ROOT_DIR`.

<pre>
<code>
📂 PhysicalAI-GR00T-Tuned-Tasks
├── exhaust_pipe_sorting_task.hdf5
├── 📂 nut_pouring_task
│   └── 📂 lerobot
│       ├── 📂 data
│       │   └── chunk-000
│       ├── 📂 meta
│       │   ├── episodes.jsonl
│       │   ├── info.json
│       │   ├── modality.json
│       │   └── tasks.jsonl
│       └── 📂videos
│           └── chunk-000
├── nut_pouring_task.hdf5
└── README.md
</code>
</pre>

#### Adapting to other embodiments & datasets

During data collection, the lower body of the GR1 humanoid is fixed, and the upper body performs tabletop manipulation
tasks. The ordered sets of joints observed in simulation ([i.e. robot states from Issac Lab](scripts/config/gr1/state_joint_space.yaml)) and commanded in simulation ([i.e. robot actions from Issac Lab](scripts/config/gr1/action_joint_space.yaml)) are included. During policy post-training and inference, only non-mimic joints in the upper body, i.e. arms and hands, are captured by the policy's observations and predictions. The ordered set of joints observed and commanded in policy ([i.e. robot joints from GR00T N1](scripts/config/gr00t/gr00t_joint_space.yaml)) are specified for data conversion remapping.

GR00T-Lerobot schema also requires [additional metadata](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md#meta). We include as them ([info.json](scripts/config/gr00t/info.json), [modality.json](scripts/config/gr00t/info.json)) as templates to faciliate conversion. If you are working with other embodiments and data configurations, please modify them accordingly.

If you are interested in leveraging this tool for other tasks, please change the task metadata in `EvalTaskConfig' defined in the [configuration] (scripts/config/args.py). More manipulation tasks are coming soon!

### Post-training

We finetuned the pre-trained [GR00T-N1-2B policy](https://huggingface.co/nvidia/GR00T-N1-2B) on these two task-specific datasets. We provided the configurations with which we obtained the above checkpoints. With one node of H100s,

```bash
python scripts/gr00t_finetune.py \
    --dataset_path=${DATASET_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --data_config=gr1_arms_only \
    --batch_size=96 \
    --max_steps=20000 \
    --num_gpus=8 \
    --save_steps=5000 \
    --base_model_path=nvidia/GR00T-N1-2B \
    --no_tune_llm  \
    --tune_visual \
    --tune_projector \
    --tune_diffusion_model \
    --no-resume \
    --dataloader_num_workers=16 \
    --report_to=wandb \
    --embodiment_tag=gr1
```
💡 **Tip:**

1. Tuning with visual backend, action projector and diffusion model generally yields smaller trajectories errors (MSE), and higher closed-loop success rates.

2. If you prefer tuning with less powerful GPUs, please follow the [reference guidelines](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#3-fine-tuning) about other finetuning options.


## 📦 Downloading Checkpoints

We post-trained the Isaac GR00T N1 policy using the above dataset, and the finetuned checkpoints are available to download.

- [GR00T-N1-2B-tuned-Nut-Pouring-task](https://huggingface.co/nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task)
- [GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task](https://huggingface.co/nvidia/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task)

Make sure you have registered your Hugging Face account and have read-access token ready.
```bash
# Provide your access token with read permission
huggingface-cli login

export CKPT="nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task"
# Or, to use the other checkpoint, uncomment the next line:
# export CKPT="nvidia/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting"
huggingface-cli download --resume-download $CKPT --local-dir
```

## 📈 Policy Closed-loop Evaluation

You can deploy the post-trained GR00T N1 policy for closed-loop control of the GR1 robot within an Issac Lab environment, and benchmark its success rate in paralle runs.

### Benchmarking Features

#### 🚀 Parallelized Evaluation:
Isaac Lab supports parallelized environment instances for scalable benchmarking. Configure multiple parallel runs (e.g., 10–100 instances) to statistically quantify policy success rates under varying initial conditions.

<table>
  <tr>
    <td align="center">
      <img src="doc/gr-1_gn1_tuned_nut_pourin.gif" width="400"/><br>
      <b>Nut Pouring</b>
    </td>
    <td align="center">
      <img src="doc/gr-1_gn1_tuned_exhaust_pipe.gif" width="400"/><br>
      <b>Exhaust Pipe Sorting</b>
    </td>
  </tr>
</table>

#### ✅ Success Metrics:
- Task Completion: Binary success/failure based on object placement accuracy defined in the [evaluation tasks](#️-evaluation-tasks). Success rates are logged in the teriminal per episode as,

```bash
==================================================
Successful trials: 9, out of 10 trials
Success rate: 0.9
==================================================
```
And the summary report as json file can be viewed as,
<pre>
{
    "metadata": {
        "checkpoint_name": "gr00t-n1-2b-tuned",
        "seed": 10,
        "date": "2025-05-20 16:42:54"
    },
    "summary": {
        "successful_trials": 91,
        "total_rollouts": 100,
        "success_rate": 0.91
    }
</pre>

To run parallel evaluation on the Nut Pouring task:

```bash
# export EVAL_RESULTS_FNAME="./eval_nutpouring.json"
python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name nutpouring \
    --eval_file_path $EVAL_RESULTS_FNAME \
# Assume the post-trained policy checkpoints are under CKPTS_PATH
    --model_path $CKPTS_PATH \
    --rollout_length 30 \
    --seed 10 \
    --max_num_rollouts 100
```

To run parallel evaluation on the Exhaust Pipe Sorting task:

```bash
# export EVAL_RESULTS_FNAME="./eval_pipesorting.json"
python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name pipesorting \
    --eval_file_path $EVAL_RESULTS_FNAME \
    --checkpoint_name gr00t-n1-2b-tuned-pipesorting \
# Assume the post-trained policy checkpoints are under CKPTS_PATH
    --model_path $CKPTS_PATH \
    --rollout_length 20 \
    --seed 10 \
    --max_num_rollouts 100
```

We report the success rate of evaluating tuned GR00T N1 policy over 200 trials, with random seed=15.

| Evaluation Task      | SR       |
|----------------------|----------|
| Nut Pouring          | 91%      |
| Exhaust Pipe Sorting | 95%      |

💡 **Tip:**
1. Hardware requirement: Please follow the system requirements in [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements) and [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#3-fine-tuning) to choose. The above evaluation results was reported on RTX A6000 Ada, Ubuntu 22.04.

2. `num_feedback_actions` determines the number of feedback actions to execute per inference, and it can be less than `action_horizon`. This option will impact the success rate of evaluation task even with the same checkpoint.

3. `rollout_length` impacts how many batched inference to make before task termination. Normally we set it between 20 to 30 for a faster turnaround.

4. `num_envs` decides the number of environments to run in parallel. Increase it too much (e.g. >100 on RTX A6000 Ada) will significantly slow down the UI rendering. We recommend set between 10 to 30 for smooth rendering and efficient benchmarking.

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

#### Pip package version mismatch

If you observe any of the following during [installation of GR00T](#-installation), you can ignore those errors.
The GR00T policy runs on an older version of torch library with flash attention, and all other tools in this repository do not require
torch>=2.7. Thus we downgrade the torch and related softwares to support GR00T inference. Mimic-related data generation workflows are not impacted.
<pre>
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
isaaclab 0.37.2 requires trimesh, which is not installed.
dex-retargeting 0.4.6 requires lxml>=5.2.2, which is not installed.
dex-retargeting 0.4.6 requires trimesh>=4.4.0, which is not installed.
isaaclab-tasks 0.10.31 requires torch>=2.7, but you have torch 2.5.1 which is incompatible.
isaacsim-kernel 5.0.0 requires wrapt==1.16.0, but you have wrapt 1.14.1 which is incompatible.
isaaclab-rl 0.2.0 requires pillow==11.0.0, but you have pillow 11.2.1 which is incompatible.
isaaclab-rl 0.2.0 requires torch>=2.7, but you have torch 2.5.1 which is incompatible.
isaaclab 0.37.2 requires pillow==11.0.0, but you have pillow 11.2.1 which is incompatible.
isaaclab 0.37.2 requires starlette==0.46.0, but you have starlette 0.45.3 which is incompatible.
isaaclab 0.37.2 requires torch>=2.7, but you have torch 2.5.1 which is incompatible.
isaacsim-core 5.0.0 requires torch==2.7.0, but you have torch 2.5.1 which is incompatible.
</pre>

#### Running on Blackwell GPUs

Unfortunately, due to limited support of flash attention module (by May 2025), GR00T policy can only support running on non-Blackwell GPUs. However
you can run Mimic-related data generation workflows and GR00T-Lerobot data conversion on Blackwell. Blackwell support is coming soon.

#### Running evaluation on Multiple GPUs

For rendering, please refer to the [Omniverse Devloper Guideline](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q9-how-to-specify-what-gpus-to-run-omniverse-apps-on) for setting single-gpu mode or multi-gpu mode of Isaac Sim. For physics, we suggest to the evaluation to run on CPU
set by `simulation_device` in evaluation.

However, GR00T N1 policy only supports single-GPU inference (by May 2025). We have not tested on multi-GPU inference.
