#!/bin/bash
# OpenWebText NCE EBM finetuning (token-additive readout + remask-ready sampler config).
# Requests 4x A100 80GB across 2 nodes (2 GPUs/node). Run after an MDLM checkpoint is available.
# Partition notes:
#   - gpuq: non-pre-emptible
#   - contrib-gpuq: same nodes, may be pre-empted (often faster queue time)
# `sgpu` shows availability: need enough free A100.80gb to place 2 GPUs on each of 2 nodes.
#
# Submit examples:
#   sbatch scripts/job_owt_nce_readout_remask_2gpu.slurm
#   sbatch --export=NONE,MDLM_CKPT=/abs/path/to/your_diffusion.ckpt scripts/job_owt_nce_readout_remask_2gpu.slurm
#
# Diffusion source for initialization:
#   - Default is Hugging Face repo id: kuleshov-group/mdlm-owt
#   - You can also set MDLM_CKPT to a local .ckpt path or local HF model directory.
#
#SBATCH --job-name=owt_nce_readout_remask
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --output=owt_nce_readout_remask-%j.out
#SBATCH --error=owt_nce_readout_remask-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=amd
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path="${SCRATCH}/edlm"
export MDLM_CKPT="${MDLM_CKPT:-kuleshov-group/mdlm-owt}"
export EXP_NAME="${EXP_NAME:-owt_nce_readout_remask}"
export CKPT_RUN_DIR="${path}/checkpoints/openwebtext/${EXP_NAME}"

mkdir -p "${path}"
cd "${path}" || exit 1

if [ ! -d "Energy-Diffusion-LLM" ]; then
  git clone https://github.com/tgiles4/Energy-Diffusion-LLM.git
fi
cd Energy-Diffusion-LLM || exit 1

if [[ "${MDLM_CKPT}" == *.ckpt ]]; then
  if [ ! -f "${MDLM_CKPT}" ]; then
    echo "ERROR: MDLM .ckpt not found: ${MDLM_CKPT}"
    echo "Set MDLM_CKPT when submitting, e.g.:"
    echo "  sbatch --export=NONE,MDLM_CKPT=/abs/path/to/your_diffusion.ckpt scripts/job_owt_nce_readout_remask_2gpu.slurm"
    exit 2
  fi
elif [ -d "${MDLM_CKPT}" ]; then
  echo "Using local HF model directory: ${MDLM_CKPT}"
else
  echo "Using Hugging Face model id: ${MDLM_CKPT}"
fi

# Load modules
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV="${SCRATCH}/edlm/venv"
export HYDRA_FULL_ERROR=1
# Optional: reduce allocator fragmentation on long runs (see PyTorch CUDA memory notes).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# W&B key: optional, loaded from a local file if present.
# Create once:
#   echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key
#   chmod 600 ~/.wandb_api_key
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

# EBM finetune from diffusion checkpoint with token-additive readout.
# Remask is a sampler-time feature (not part of NCE loss), but we set it here
# so sample/eval behavior from this run can use the new path without extra overrides.
#
# Batch layout (2 nodes x 2 GPUs, DDP): dataloader asserts
#   global_batch_size == batch_size * num_gpus * accumulate_grad_batches
# Per-GPU microbatch 32, grad accumulation is derived from
#   loader.global_batch_size / (trainer.num_nodes * trainer.devices * loader.batch_size).
# Val builds full log_p_x0 [B,T,V] (~50k vocab) + exp + Gumbel noise — memory ~linear in
# eval batch per GPU. With eval_global_batch_size=128 on 4 GPUs total, per-GPU eval batch is 32.
# Checkpointing is pinned to a stable run dir so interrupted jobs resume from last.ckpt.
srun --ntasks=4 --ntasks-per-node=2 --export=ALL "${ENV}/bin/python" -u -m main \
  ++path="${path}" \
  train_mdlm_only=false \
  data=openwebtext \
  data.valid=openwebtext-valid \
  model=small \
  model.length=1024 \
  model.dropout=0.1 \
  eval.checkpoint_path="${MDLM_CKPT}" \
  ebm_backbone=hf_dit \
  ebm_readout=token_additive \
  sampling.predictor=ddpm_cache \
  sampling.ebm_remask=true \
  sampling.is_start=0.6 \
  sampling.is_end=0.4 \
  sampling.ebm_remask_weight_eps=1e-8 \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_global_batch_size=128 \
  loader.num_workers=0 \
  trainer.num_nodes=2 \
  trainer.max_steps=400000 \
  trainer.val_check_interval=1.0 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2500 \
  callbacks.checkpoint_monitor.monitor=val/nll \
  callbacks.checkpoint_monitor.save_top_k=3 \
  callbacks.checkpoint_monitor.save_last=true \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=125 \
  checkpointing.run_dir="${CKPT_RUN_DIR}" \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="${CKPT_RUN_DIR}/last.ckpt" \
  hydra.run.dir=outputs/owt_nce_readout_remask \
  wandb.id=null \
  wandb.group=owt_nce_readout_remask \
  wandb.name=owt_nce_readout_remask
