# Post Training

[GR00T N1](https://github.com/NVIDIA/Isaac-GR00T/tree/n1-release?tab=readme-ov-file#nvidia-isaac-gr00t-n1) is a foundation model for generalized humanoid robot reasoning and skills, trained on an extensive multimodal dataset that includes real-world, synthetic, and internet-scale data. The model is designed for cross-embodiment generalization and can be efficiently adapted to new robot embodiments, tasks, and environments through post training.

We followed the recommended GR00T N1 post training workflow to adapt the model for the Fourier GR1 robot, targeting two industrial manipulation tasks: nut pouring and exhaust pipe sorting. The process involves multiple steps introduced below. You can also skip to the next section [Downloading Checkpoints](#downloading-checkpoints) to get post-trained checkpoints.

## Data Conversion

The process involved converting demonstration data (Mimic-generated motion trajectories in HDF5) into the LeRobot-compatible schema ([GR00T-Lerobot format guidelines](https://github.com/NVIDIA/Isaac-GR00T/blob/n1-release/getting_started/LeRobot_compatible_data_schema.md)).


- Using a python interpreter or conda/virtual env that has Isaac Lab, GR00T and Eavluation Tasks installed, convert Mimic-generated trajectories by

```bash
# Example: Set `task_name` Based on Task
# Nut Pouring
export TASK_NAME="nutpouring"
# Uncomment the below when Task is Exhaust Pipe Sorting
# export TASK_NAME="pipesorting"

# Within IsaacLabEvalTasks directory
# DATASET_ROOT_DIR is directory of where Mimic-generated HDF5 is saved locally
python scripts/convert_hdf5_to_lerobot.py --task_name $TASK_NAME --data_root $DATASET_ROOT_DIR
```

The GR00T-LeRobot-compatible datasets will be available in `DATASET_ROOT_DIR`.

<pre>
<code>
📂 PhysicalAI-GR00T-Tuned-Tasks
├── exhaust_pipe_sorting_task.hdf5
├── 📂 nut_pouring_task
│   └── 📂 lerobot
│       ├── 📂 data
│       │   └── chunk-000
│       ├── 📂 meta
│       │   ├── episodes.jsonl
│       │   ├── info.json
│       │   ├── modality.json
│       │   └── tasks.jsonl
│       └── 📂videos
│           └── chunk-000
├── nut_pouring_task.hdf5
└── README.md
</code>
</pre>

### Adapting to other embodiments & datasets

During data collection, the lower body of the GR1 humanoid is fixed, and the upper body performs tabletop manipulation
tasks. The ordered sets of joints observed in simulation ([i.e. robot states from Isaac Lab](scripts/config/gr1/state_joint_space.yaml)) and commanded in simulation ([i.e. robot actions from Isaac Lab](scripts/config/gr1/action_joint_space.yaml)) are included. During policy post training and inference, only non-mimic joints in the upper body, i.e. arms and hands, are captured by the policy's observations and predictions. The ordered set of joints observed and commanded in policy ([i.e. robot joints from GR00T N1](scripts/config/gr00t/gr00t_joint_space.yaml)) are specified for data conversion remapping.

GR00T-Lerobot schema also requires [additional metadata](https://github.com/NVIDIA/Isaac-GR00T/blob/n1-release/getting_started/LeRobot_compatible_data_schema.md#meta). We include them ([info.json](scripts/config/gr00t/info.json), [modality.json](scripts/config/gr00t/info.json)) as templates to facilitate conversion. If you are working with other embodiments and data configurations, please modify them accordingly.

If you are interested in leveraging this tool for other tasks, please change the task metadata in `EvalTaskConfig` defined in the [configuration](scripts/config/args.py). The `TASK_NAME` is associated with the pre-defined task description in [`Gr00tN1DatasetConfig`](scripts/config/args.py) class. The task_index indicates the index associated with language description, and 1 is reserved for data validity check, following GR00T-N1 guidelines. You may want to add other indices for your self-defined task. More manipulation tasks are coming soon!

## Post Training

We finetuned the pre-trained [GR00T-N1-2B policy](https://huggingface.co/nvidia/GR00T-N1-2B) on these two task-specific datasets. We provided the configurations with which we obtained the above checkpoints. With one node of H100s,

```bash
# Within IsaacLabEvalTasks directory
cd submodules/Isaac-GR00T
# Provide the directory where the GR00T-Lerobot data is stored as DATASET_PATH
# Please use full path, instead of relative path
# Nut pouring
# E.g. export DATASET_PATH=/home/data/PhysicalAI-GR00T-Tuned-Tasks/nut_pouring_task/lerobot
# Exhaust pipe sorting
# E.g. export DATASET_PATH=/home/data/PhysicalAI-GR00T-Tuned-Tasks/Exhaust-Pipe-Sorting-task/lerobot
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

2. If you prefer tuning with less powerful GPUs, please follow the [reference guidelines](https://github.com/NVIDIA/Isaac-GR00T/tree/n1-release?tab=readme-ov-file#3-fine-tuning) about other finetuning options.
