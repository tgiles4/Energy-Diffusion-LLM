#!/bin/bash
# OpenWebText EBM training from MDLM checkpoint.
# Requests 4x A100 80GB across 2 nodes (2 GPUs/node).
# Partition notes:
#   - gpuq: non-pre-emptible
#   - contrib-gpuq: same nodes, may be pre-empted (often faster queue time)
#
# Submit examples:
#   sbatch scripts/job_train_owt_ebm.sh
#   sbatch --export=NONE,MDLM_CKPT=/abs/path/to/your_diffusion.ckpt scripts/job_train_owt_ebm.sh
#
# Diffusion source for initialization:
#   - Default is Hugging Face repo id: kuleshov-group/mdlm-owt
#   - You can also set MDLM_CKPT to a local .ckpt path or local HF model directory.
#
#SBATCH --job-name=owt_ebm_train
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --output=owt_ebm_train-%j.out
#SBATCH --error=owt_ebm_train-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=amd
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path="${SCRATCH}/edlm"
export MDLM_CKPT="${MDLM_CKPT:-kuleshov-group/mdlm-owt}"
export EXP_NAME="${EXP_NAME:-ebm_owt_load_hf}"
export CKPT_SAVE_DIR="${path}/openwebtext/${EXP_NAME}"

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
    echo "  sbatch --export=NONE,MDLM_CKPT=/abs/path/to/your_diffusion.ckpt scripts/job_train_owt_ebm.sh"
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

# W&B key: optional, loaded from a local file if present.
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

srun --ntasks=4 --ntasks-per-node=2 --export=ALL "${ENV}/bin/python" -u -m main \
  ++path="${path}" \
  train_mdlm_only=false \
  data=openwebtext \
  data.valid=openwebtext-valid \
  loader.batch_size=32 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=128 \
  model=small \
  model.dropout=0.1 \
  wandb.name="${EXP_NAME}" \
  hydra.run.dir="outputs/${EXP_NAME}" \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=true \
  sampling.steps=1000 \
  backbone=hf_dit \
  eval.checkpoint_path="${MDLM_CKPT}" \
  sampling.num_sample_batches=1 \
  trainer.num_nodes=2 \
  trainer.max_steps=400000 \
  trainer.val_check_interval=0.25 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2500 \
  checkpointing.save_dir="${CKPT_SAVE_DIR}" \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="${CKPT_SAVE_DIR}/checkpoints/last.ckpt"