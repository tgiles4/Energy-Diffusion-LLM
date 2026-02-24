# Running on a Slurm cluster (home + scratch, Text8)

This project is run on a cluster with **home** (persistent, 60 GB quota) and **scratch** (large, wiped every 90 days). Experiments use **Text8** only.

- **Paths**: `$HOME=/home/tgiles`, `$SCRATCH=/scratch/tgiles`
- **Slurm**: `sbatch`, `srun` at `/usr/bin`

Cluster-specific guides (PDFs) can be placed in this directory, e.g.:
- `docs/slurm-user-guide.pdf`
- `docs/FY25-ClusterQuickStartGuide.pdf`

## Storage strategy

| Location | Use | Notes |
|----------|-----|--------|
| **Home** (~60 GB) | Repo, Slurm scripts, conda env (if it fits), long-term checkpoints | Copy important checkpoints from scratch before 90-day wipe. |
| **Scratch** | Job run dir, data cache, outputs, checkpoints during run | Set job `cwd` (or `path`) to `$SCRATCH/edlm` or `$SCRATCH/$SLURM_JOB_ID`. |

- **Text8**: Data is small (~100 MB raw). Cache under `path/data/text8` is fine on scratch.
- **Checkpoints**: Written to `checkpointing.save_dir` (defaults to `${cwd}`). Run with cwd on scratch, then copy `checkpoints/` (and any `outputs/`) to home for anything you need beyond 90 days.

## Login node vs compute nodes

**Login node** and **compute nodes** are different. Your job runs on **compute nodes**; the login node is only for submitting jobs, editing files, and light checks.

| | Login node | Compute nodes (where the job runs) |
|--|------------|-------------------------------------|
| **Use** | Submit jobs, clone/edit code, run `module avail` | Actual training; request 4× A100 here |
| **Python / Conda** | May be old or not in PATH (e.g. Python 3.6.8) | Set up in the **Slurm script** so the job gets Python 3.9+ and your env |
| **GPUs** | May show a small GPU (e.g. A2) or none | Request 4× A100 (40 or 80 GB) via Slurm directives |

Your Slurm script must load modules and activate the conda env **inside the script**, so that when the job starts on a compute node, it has the right Python and CUDA. **Modules loaded on the login node are not available on compute nodes** — each job gets a fresh environment, so load what you need in the job script. If your cluster has different software on login vs compute, run `module avail` (and optionally `python3 --version`, `nvidia-smi`) **on a compute node** to confirm, e.g.:

```bash
srun --partition=YOUR_PARTITION --gres=gpu:1 --pty bash -l
# then: module avail 2>&1 | grep -i anaconda
```

Replace `YOUR_PARTITION` with the partition that has A100s (see the cluster guide).

## Modules (LMod)

The cluster uses **LMod** (Lua-based modules). Only modules compatible with your currently loaded compiler/MPI are shown — so you may need to load a **base** (e.g. compiler) before Python/Anaconda appears in `module avail`.

Useful commands:
- `module` — usage and options
- `module avail` — what modules are available (depends on what’s already loaded)
- `module list` — what’s currently loaded
- `module load <module_name>` — load a module (must do this before using that software)

If `module avail` doesn’t show Anaconda/Python, load the compiler (or other base) first as in the cluster guide (e.g. `gnu9/9.3.0`), then run `module avail` again. In your **Slurm script**, load any required base modules before loading Miniconda so the environment is correct on compute nodes.

## Environment (Conda)

This project requires **Python 3.9+**. On Hopper, module Python goes up to 3.8; use **Miniconda3** to create an env with Python 3.9 and the project deps.

**Available on Hopper (GNU-9.3.0_OpenMPI-4.0.4):** `miniconda3/4.9.2-vl` (no Anaconda; Python modules are 2.7, 3.7, 3.8 only).

1. Load Miniconda and create the project env once (on login or on a compute node):
   ```bash
   module load miniconda3/4.9.2-vl
   conda env create -f requirements.yaml   # from repo root; creates env with python=3.9
   # or: conda create -n edlm python=3.9 && conda activate edlm && pip install ...
   ```
   Create the env on home or, if space is tight, under `$SCRATCH/edlm/env` and activate that in jobs.
2. In your **Slurm job script** (executed on compute nodes), load the same module and activate the env:
   ```bash
   # module load gnu9/9.3.0    # uncomment if your partition requires it
   module load miniconda3/4.9.2-vl
   conda activate edlm
   ```
   Load any base modules (e.g. `gnu9/9.3.0`) first if your partition or the cluster guide requires it.

## Path and config

Scripts and configs expect a base **path** used for:
- Shell: `cd ${path}` (job working directory).
- Hydra: `data.cache_dir` = `${path}/data` (e.g. `$SCRATCH/edlm/data`).

In your Slurm script, set once and use everywhere, e.g.:

```bash
export path=${SCRATCH}/edlm   # or ${SCRATCH}/${SLURM_JOB_ID} for per-job dirs
cd ${path}
# Optional: clone/copy repo here, or run from home and only set path for data/checkpoints
```

Override for Hydra so the config sees the same path:

```bash
python -u -m main path=${path} data=text8 model.length=256 ...
```

## Text8-specific overrides

- Use `data=text8` (or `data=text8-crop`).
- Configs suggest `model.length=256` for Text8 (see `configs/data/text8.yaml`). Override: `model.length=256`.

Example minimal run (conceptually; actual Slurm directives go in the job script):

```bash
python -u -m main \
  path=${path} \
  data=text8 \
  model.length=256 \
  hydra.run.dir=outputs/text8_run \
  checkpointing.save_dir=${path} \
  ...
```

## After a run

To keep checkpoints and logs past 90 days, copy from scratch to home before the wipe, e.g.:

```bash
cp -r ${path}/checkpoints ${HOME}/edlm-checkpoints/
cp -r ${path}/outputs    ${HOME}/edlm-outputs/
```
