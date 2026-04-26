#!/bin/bash
# Validate OpenWebText NCE EBM checkpoint on the default validation set (wikitext103).
# Designed to evaluate checkpoints produced by:
#   scripts/job_owt_nce_readout_remask_2gpu.slurm
#
# Submit examples:
#   sbatch scripts/job_owt_nce_readout_remask_val_2gpu.slurm
#   sbatch --export=NONE,NCE_CKPT=/abs/path/to/checkpoint.ckpt scripts/job_owt_nce_readout_remask_val_2gpu.slurm
#
# Default checkpoint path matches training script output:
#   ${SCRATCH}/edlm/checkpoints/last-v5.ckpt
#
#SBATCH --job-name=owt_nce_readout_val
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --output=owt_nce_readout_val-%j.out
#SBATCH --error=owt_nce_readout_val-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=amd
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path="${SCRATCH}/edlm"
export NCE_CKPT="${NCE_CKPT:-${path}/checkpoints/last-v5.ckpt}"

mkdir -p "${path}"
cd "${path}" || exit 1

if [ ! -d "Energy-Diffusion-LLM" ]; then
  git clone https://github.com/tgiles4/Energy-Diffusion-LLM.git
fi
cd Energy-Diffusion-LLM || exit 1

if [ ! -f "${NCE_CKPT}" ]; then
  echo "ERROR: Checkpoint not found: ${NCE_CKPT}"
  echo "Set NCE_CKPT when submitting, e.g.:"
  echo "  sbatch --export=NONE,NCE_CKPT=/abs/path/to/checkpoint.ckpt scripts/job_owt_nce_readout_remask_val_2gpu.slurm"
  exit 2
fi

# Load modules
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV="${SCRATCH}/edlm/venv"
export HYDRA_FULL_ERROR=1
# Optional: reduce allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# W&B key: optional, loaded from local file if present.
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

# Validation-only run using the NCE checkpoint.
# data=openwebtext keeps default valid split (wikitext103) from configs/data/openwebtext.yaml.
# Keep per-GPU eval batch at 32 (eval_global_batch_size=64 across 2 GPUs).
srun --ntasks=2 --ntasks-per-node=2 --export=ALL "${ENV}/bin/python" -u -m main \
  ++path="${path}" \
  mode=ppl_eval \
  train_mdlm_only=false \
  data=openwebtext \
  model=small \
  model.length=1024 \
  model.dropout=0.1 \
  ebm_backbone=hf_dit \
  ebm_readout=token_additive \
  sampling.predictor=ddpm_cache \
  sampling.ebm_remask=true \
  sampling.ebm_use_grad_contrib=true \
  sampling.is_start=0.6 \
  sampling.is_end=0.4 \
  sampling.ebm_remask_weight_eps=1e-8 \
  eval.checkpoint_path="${NCE_CKPT}" \
  loader.eval_global_batch_size=64 \
  loader.num_workers=0 \
  trainer.num_nodes=1 \
  checkpointing.save_dir="${path}" \
  hydra.run.dir=outputs/owt_nce_readout_remask_val \
  wandb.id=null \
  wandb.group=owt_nce_readout_remask_val \
  wandb.name=owt_nce_readout_remask_val
