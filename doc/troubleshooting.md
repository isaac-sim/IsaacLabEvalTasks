# Troubleshooting

## Pip package version mismatch

If you observe any of the following during [installation of GR00T](installation.md), you can ignore those errors.
The GR00T policy runs on an older version of torch library with flash attention, and all other tools in this repository do not require
torch>=2.7. Thus we downgrade the torch and related software to support GR00T inference. Mimic-related data generation workflows are not impacted.
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

## Running on Blackwell GPUs

Unfortunately, due to limited support of flash attention module (by May 2025), GR00T policy can only support running on non-Blackwell GPUs. However
you can run Mimic-related data generation workflows and GR00T-Lerobot data conversion on Blackwell. Blackwell support is coming soon.

## Running evaluation on Multiple GPUs

For rendering, please refer to the [Omniverse Developer Guideline](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q9-how-to-specify-what-gpus-to-run-omniverse-apps-on) for setting single-gpu mode or multi-gpu mode of Isaac Sim. For physics, we suggest to the evaluation to run on CPU
set by `simulation_device` in evaluation.

However, GR00T N1 policy only supports single-GPU inference (by May 2025). We have not tested on multi-GPU inference.
