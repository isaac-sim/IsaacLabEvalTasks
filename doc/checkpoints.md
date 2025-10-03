# Downloading Checkpoints

We post-trained the Isaac GR00T N1 policy using the above dataset, and the finetuned checkpoints are available to download.

## Available Checkpoints

- [GR00T-N1-2B-tuned-Nut-Pouring-task](https://huggingface.co/nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task)
- [GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task](https://huggingface.co/nvidia/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task)

## Download Instructions

Make sure you have registered your Hugging Face account and have read-access token ready.

```bash
# Provide your access token with read permission
huggingface-cli login

export CKPT="nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task"
# Or, to use the other checkpoint, uncomment the next line:
# export CKPT="nvidia/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task"
# Define the path to save the checkpoints as CKPT_LOCAL_DIR
huggingface-cli download --resume-download $CKPT --local-dir $CKPT_LOCAL_DIR
```
