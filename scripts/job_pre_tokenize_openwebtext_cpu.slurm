#!/bin/bash
# One-off: build OpenWebText + WikiText-103 tokenized caches under ${path}/data (CPU).
# Run this before GPU training so the training job skips the long "Tokenizing" phase.
#
# Cluster-aware resources (one full CPU node; preprocess_num_proc = cpus Slurm grants):
#   Intel:  48 cores / 192 GB  ->  --cpus-per-task=48   --mem-per-cpu=4GB
#   AMD:    64 cores / 256 GB  ->  --cpus-per-task=64   --mem-per-cpu=4GB   (default below)
#   AMD:   128 cores / 2–4 TB  ->  --cpus-per-task=128  --mem-per-cpu=2GB   (256 GB; or raise GB if OOM)
# Requesting more cores than a node has prevents scheduling; 128 cores only lands on 128-core nodes.
# If OOM during map: raise mem-per-cpu or lower data.preprocess_map_batch_size.
#
# Submit:
#   sbatch scripts/job_pre_tokenize_openwebtext_cpu.slurm
#   sbatch --export=NONE,path=/scratch/$USER/my_edlm scripts/job_pre_tokenize_openwebtext_cpu.slurm

#SBATCH --partition=normal
#SBATCH --job-name=edlm_pre_tok
#SBATCH --output=edlm_pre_tokenize-%j.out
#SBATCH --error=edlm_pre_tokenize-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=0-08:00:00

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path="${path:-${SCRATCH}/edlm}"
export HYDRA_FULL_ERROR=1
# Avoid tokenizer/fork spam when datasets uses many worker processes.
export TOKENIZERS_PARALLELISM=false
# Optional: faster Hub downloads after `pip install hf_transfer` in the venv.
# export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "${path}"
cd "${path}" || exit 1

if [ ! -d "Energy-Diffusion-LLM" ]; then
  git clone https://github.com/tgiles4/Energy-Diffusion-LLM.git
fi
cd Energy-Diffusion-LLM || exit 1

# Site-specific Python stack (adjust for your cluster).
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh

ENV="${SCRATCH}/edlm/venv"

# Same data layout as scripts/job_owt_nce_readout_remask_2gpu.slurm (openwebtext + wikitext103 valid).
"${ENV}/bin/python" -u pre_tokenize_data.py \
  hydra.job.chdir=false \
  hydra.run.dir="${path}/outputs/hydra_pre_tokenize" \
  ++path="${path}" \
  data=openwebtext \
  model=small \
  model.length=1024
