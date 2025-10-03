# Policy Closed-loop Evaluation

You can deploy the post-trained GR00T N1 policy for closed-loop control of the GR1 robot within an Isaac Lab environment, and benchmark its success rate in parallel runs.

## Benchmarking Features

### 🚀 Parallelized Evaluation:
Isaac Lab supports parallelized environment instances for scalable benchmarking. Configure multiple parallel runs (e.g., 10–100 instances) to statistically quantify policy success rates under varying initial conditions.

<table>
  <tr>
    <td align="center">
      <img src="gr-1_gn1_tuned_nut_pourin.gif" width="400"/><br>
      <b>Nut Pouring</b>
    </td>
    <td align="center">
      <img src="gr-1_gn1_tuned_exhaust_pipe.gif" width="400"/><br>
      <b>Exhaust Pipe Sorting</b>
    </td>
  </tr>
</table>

### ✅ Success Metrics:
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

## Running Evaluation

### Nut Pouring Task

To run parallel evaluation on the Nut Pouring task:

```bash
# Within IsaacLabEvalTasks directory
# Assume the post-trained policy checkpoints are under CKPTS_PATH
# Please use full path, instead of relative path for CKPTS_PATH
# export EVAL_RESULTS_FNAME="./eval_nutpouring.json"
python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name nutpouring \
    --eval_file_path $EVAL_RESULTS_FNAME \
    --model_path $CKPTS_PATH \
    --rollout_length 30 \
    --seed 10 \
    --max_num_rollouts 100
```

### Exhaust Pipe Sorting Task

To run parallel evaluation on the Exhaust Pipe Sorting task:

```bash
# Assume the post-trained policy checkpoints are under CKPTS_PATH
# Please use full path, instead of relative path for CKPTS_PATH
# export EVAL_RESULTS_FNAME="./eval_pipesorting.json"
python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name pipesorting \
    --eval_file_path $EVAL_RESULTS_FNAME \
    --checkpoint_name gr00t-n1-2b-tuned-pipesorting \
    --model_path $CKPTS_PATH \
    --rollout_length 20 \
    --seed 10 \
    --max_num_rollouts 100
```

## Performance Results

We report the success rate of evaluating tuned GR00T N1 policy over 200 trials, with random seed=15.

| Evaluation Task      | SR       |
|----------------------|----------|
| Nut Pouring          | 91%      |
| Exhaust Pipe Sorting | 95%      |

## Tips and Best Practices

💡 **Tip:**
1. Hardware requirement: Please follow the system requirements in [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements) and [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/n1-release?tab=readme-ov-file#3-fine-tuning) to choose. The above evaluation results was reported on RTX A6000 Ada, Ubuntu 22.04.

2. `num_feedback_actions` determines the number of feedback actions to execute per inference, and it can be less than `action_horizon`. This option will impact the success rate of evaluation task even with the same checkpoint.

3. `rollout_length` impacts how many batched inference to make before task termination. Normally we set it between 20 to 30 for a faster turnaround.

4. `num_envs` decides the number of environments to run in parallel. Too many parallel environments (e.g. >100 on RTX A6000 Ada) will significantly slow down the UI rendering. We recommend to set between 10 to 30 for smooth rendering and efficient benchmarking.
