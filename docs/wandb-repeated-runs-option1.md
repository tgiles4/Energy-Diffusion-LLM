# W&B repeated runs (Option 1)

Run the same experiment multiple times and log each as a **separate trendline** in Weights & Biases, without changing config or code.

## Secure API key on a SLURM cluster

The job scripts (`scripts/job_text8_mdlm_4gpu.sh`, `scripts/job_text8_nce_4gpu.sh`) log to W&B but do not contain your token. They read **`WANDB_API_KEY`** from a file so the key is never in the script or in `sbatch --export` (which can be visible in the process list).

1. **One-time setup** (on the login node or any node; home is usually shared so all compute nodes see it):
   ```bash
   echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key
   chmod 600 ~/.wandb_api_key
   ```
2. The scripts check for `~/.wandb_api_key` and, if present, set `export WANDB_API_KEY=$(cat ~/.wandb_api_key)` before running. No need to know which nodes you’ll get—every node that runs your job can read your home directory.
3. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize). Never commit `~/.wandb_api_key` or put the key in the repo.

If you prefer a path under `$SCRATCH` (e.g. `$SCRATCH/.wandb_api_key`), create that file the same way and change the `if [ -f "..." ]` path in the script to match.

## What to do

1. **Override run id** so each launch creates a new run:
   ```bash
   wandb.id=null
   ```

2. **Group runs** so they appear together (e.g. mean ± std in the UI):
   ```bash
   wandb.group=my_experiment
   ```

3. Use the **same seed** and other params as usual. Launch the same command (or job script) once per trial.

## Example

```bash
python -u -m main \
    wandb.id=null \
    wandb.group=ebm_owt_10trials \
    wandb.name=ebm_owt_load_hf \
    loader.batch_size=16 \
    # ... rest of your args, same for every trial
```

Run that 10 times (manually, or submit 10 identical jobs). Each run gets a unique W&B id and shows as its own trendline under the group.

## Using this with the SLURM job scripts

To get separate trendlines when submitting multiple jobs (e.g. 10 trials), add the overrides to the `python -u -m main ...` line in the script. Example for `job_text8_mdlm_4gpu.sh`: add `wandb.id=null` and `wandb.group=text8_mdlm_10trials` alongside the existing `wandb.name=text8_mdlm`. Submit the same job script 10 times; each run will get a unique W&B id and appear under the group in the UI.

## Caveats

- Run identity is by auto-generated id/timestamp, not “trial 3” etc.
- Confirm `wandb.id=null` actually clears the id (no fixed id from config); if runs still resume, you may need a config override that omits `id` entirely.
