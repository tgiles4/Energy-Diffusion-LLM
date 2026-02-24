#!/bin/bash
# Text8 NCE EBM finetuning from MDLM checkpoint (4× A100). Run after job_text8_mdlm_4gpu.sh.
#SBATCH --job-name=text8_nce
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:4

#SBATCH --output=text8_nce-%j.out
#SBATCH --error=text8_nce-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --time=0-05:00:00

# Set path and MDLM checkpoint (run after job_text8_mdlm_4gpu.sh)
export path=${SCRATCH}/edlm
# Path to MDLM checkpoint from script 1; override when submitting if needed:
# sbatch --export=ALL,MDLM_CKPT=/path/to/checkpoints/last.ckpt job_text8_nce_4gpu.sh
export MDLM_CKPT="${MDLM_CKPT:-${path}/checkpoints/last.ckpt}"

mkdir -p ${path}
cd ${path}

if [ ! -d "Energy-Diffusion-LLM" ]; then
  git clone https://github.com/tgiles4/Energy-Diffusion-LLM.git
fi
cd Energy-Diffusion-LLM

# Load modules and conda env
# module load gnu9/9.3.0   # uncomment if your partition requires it
module load miniconda3/4.9.2-vl
conda activate edlm

# W&B: load API key from a file (avoids putting secrets in the script or sbatch env).
# Create once: echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

conda env update -f requirements.yaml --name edlm 2>/dev/null || true

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
