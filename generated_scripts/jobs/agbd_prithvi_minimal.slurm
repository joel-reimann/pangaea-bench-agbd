#!/bin/bash
#SBATCH --job-name=agbd_prithvi_minimal
#SBATCH --account=es_schin
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=400G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=./generated_scripts/logs/agbd_prithvi_minimal-%j.out
#SBATCH --error=./generated_scripts/logs/agbd_prithvi_minimal-%j.err

set -euo pipefail

echo "=========================================="
echo "AGBD Training: prithvi (minimal)"
echo "Job: ${SLURM_JOB_ID} | Started: $(date)"
echo "=========================================="

# Load modules
module load eth_proxy stack/2024-06 

# Setup workspace
TMPDIR_PANGAEA="$TMPDIR/pangaea_workspace"
TMPDIR_VENV="$TMPDIR/pangaea_venv"
TMPDIR_DATA="$TMPDIR/agbd_data"

# Copy codebase
echo "Copying codebase..."
rsync -av --exclude '__pycache__' --exclude '.git' --exclude 'wandb' --exclude 'test_agbd_logs' \
    "/cluster/home/reimannj/pangaea-bench/" "$TMPDIR_PANGAEA/"

# Copy environment  
echo "Copying environment..."
rsync -av "/cluster/home/reimannj/pangaea-bench-venv/" "$TMPDIR_VENV/"

# Copy AGBD data
echo "Copying AGBD data..."
mkdir -p "$TMPDIR_DATA"
rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/ "$TMPDIR_DATA/"

# Activate environment
cd "$TMPDIR_PANGAEA"
source "$TMPDIR_VENV/bin/activate"

# Output directory
OUTDIR="./generated_scripts/results/${SLURM_JOB_ID}_prithvi_minimal"
mkdir -p "$OUTDIR"

# Run training (EXACT SAME COMMAND AS LOCAL SCRIPT)
echo "Starting training..."
torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    pangaea/run.py \
    --config-name=train_agbd \
    dataset=agbd \
    encoder=prithvi \
    decoder=reg_upernet \
    criterion=mse \
    task=regression \
    task.trainer.n_epochs=1 \
    task.evaluator.inference_mode=whole \
    task.trainer.ckpt_interval=1 \
    task.trainer.eval_interval=1 \
    dataset.root_path="$TMPDIR_DATA" \
    dataset.hdf5_dir="$TMPDIR_DATA" \
    dataset.mapping_path="$TMPDIR_DATA" \
    dataset.norm_path="$TMPDIR_DATA" \
    seed=42 \
    use_wandb=true \
    work_dir="$OUTDIR"

echo "Training completed: $(date)"
