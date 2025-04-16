srun: job 43230 queued and waiting for resources
srun: job 43230 has been allocated resources
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: christopher-r-ellis30 (christopher-r-ellis30-carnegie-mellon-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.9
wandb: Run data is saved locally in /home/crellis/transformers2/current_progress/final_work/wandb/run-20250415_190554-5f582ywr
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run floral-blaze-5
wandb: ⭐️ View project at https://wandb.ai/christopher-r-ellis30-carnegie-mellon-university/prefix-few-shot-examples
wandb: 🚀 View run at https://wandb.ai/christopher-r-ellis30-carnegie-mellon-university/prefix-few-shot-examples/runs/5f582ywr
Loading model and tokenizer...
Model loaded...
Loaded 11 unique questions from all JSON files
Total possible combinations: 462
Sampling 100 combinations out of 462
Evaluating combinations:   0%|          | 0/100 [00:00<?, ?it/s]/home/crellis/miniconda3/envs/transformers2/lib/python3.13/site-packages/transformers/generation/configuration_utils.py:649: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
srun: forcing job termination
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
Evaluating combinations:   1%|          | 1/100 [00:37<1:02:28, 37.86s/it]slurmstepd: error: *** STEP 43230.0 ON node-gpu03 CANCELLED AT 2025-04-15T19:07:03 ***
srun: error: node-gpu03: task 0: Killed
