#!/usr/bin/env bash
#SBATCH --job-name=ecg_newdata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=46:00:00
#SBATCH --output=logs/ecg_newdata_%j.out
#SBATCH --error=logs/ecg_newdata_%j.err

set -euo pipefail
# Submit from the repository root after creating logs/. Set --account at sbatch.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${ECG_REPO_DIR:-${SLURM_SUBMIT_DIR:-$script_dir}}"
cd "$repo_dir"
python_bin="${ECG_PYTHON:-python}"
config_file="${ECG_CONFIG:-src/config/unet_a100_newdata.yml}"
run_id="${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${ECG_TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/ecg_triton_${run_id}}"
mkdir -p "$TRITON_CACHE_DIR"
nvidia-smi
exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 --module src.train --config "$config_file" "$@"
