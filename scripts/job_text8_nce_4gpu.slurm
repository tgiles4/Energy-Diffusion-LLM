#!/bin/bash
# Text8 NCE EBM finetuning from MDLM checkpoint (4× A100). Run after job_text8_mdlm_4gpu.slurm. Uses Miniconda (cluster module).
#SBATCH --job-name=text8_nce
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:4

#SBATCH --output=text8_nce-%j.out
#SBATCH --error=text8_nce-%j.err
#SBATCH --export=NONE

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=amd

#SBATCH --mem=64G
#SBATCH --time=0-05:00:00

# Set path and MDLM checkpoint (run after job_text8_mdlm_4gpu.slurm). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
# Path to MDLM checkpoint from script 1; override when submitting if needed:
# sbatch --export=NONE,MDLM_CKPT=/path/to/checkpoints/last.ckpt job_text8_nce_4gpu.slurm
export MDLM_CKPT="${MDLM_CKPT:-${path}/checkpoints/last.ckpt}"

mkdir -p ${path}
cd ${path}

if [ ! -d "Energy-Diffusion-LLM" ]; then
  git clone https://github.com/tgiles4/Energy-Diffusion-LLM.git
fi
cd Energy-Diffusion-LLM

# Load modules: compiler base (required for LMod hierarchy), Miniconda, and CUDA
module load miniconda3/4.9.2-vl

# Resolve conda prefix from PATH (provided by the module)
CONDA_EXE="$(command -v conda)"                 # .../bin/conda
CONDA_BIN="$(dirname "$CONDA_EXE")"             # .../bin
CONDA_PREFIX_ROOT="$(dirname "$CONDA_BIN")"     # ... (the conda install prefix)

# Find a working interpreter inside the prefix
if   [ -x "${CONDA_BIN}/python" ];   then CONDA_PY="${CONDA_BIN}/python"
elif [ -x "${CONDA_BIN}/python3" ];  then CONDA_PY="${CONDA_BIN}/python3"
elif [ -x "${CONDA_BIN}/python3.8" ];then CONDA_PY="${CONDA_BIN}/python3.8"
else
  echo "ERROR: No python interpreter found in ${CONDA_BIN}" >&2
  ls -l "${CONDA_BIN}/python"* >&2 || true
  exit 1
fi

# Safe conda wrapper (bypasses broken shebang)
conda() { "${CONDA_PY}" "${CONDA_EXE}" "$@"; }

# Now conda calls work
source "${CONDA_PREFIX_ROOT}/etc/profile.d/conda.sh"

# Create conda environment if it doesn't exist (Python 3.9+ required)
CONDA_ENV="${path}/edlm_env"
if [ ! -d "${CONDA_ENV}" ]; then
  conda create -y -p "${CONDA_ENV}" python=3.9
  conda activate "${CONDA_ENV}"
  conda install -y --file requirements.txt
else
  conda activate "${CONDA_ENV}"
fi

# W&B: load API key from a file (avoids putting secrets in the script or sbatch env).
# Create once: echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

# NCE EBM finetuning from MDLM: pooling + scalar energy head, 10k steps,
# cosine LR with 2000-step warmup, same optimizer settings
python -u -m main \
  path=${path} \
  train_mdlm_only=false \
  data=text8 \
  model=small \
  model.length=256 \
  model.hidden_size=784 \
  model.n_blocks=12 \
  model.n_heads=12 \
  model.dropout=0.05 \
  eval.checkpoint_path=${MDLM_CKPT} \
  ebm_backbone=dit \
  loader.global_batch_size=512 \
  trainer.max_steps=10000 \
  trainer.val_check_interval=2000 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2000 \
  checkpointing.save_dir=${path} \
  checkpointing.resume_from_ckpt=false \
  hydra.run.dir=outputs/text8_nce \
  wandb.id=null \
  wandb.group=text8_nce \
  wandb.name=text8_nce
