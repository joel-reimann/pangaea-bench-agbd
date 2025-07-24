#!/bin/bash
#SBATCH --job-name=agbd_prithvi_fast
#SBATCH --account=es_schin
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=400G
#SBATCH --gpus=1
#SBATCH --output=./generated_scripts/logs/agbd_prithvi_fast-%j.out
#SBATCH --error=./generated_scripts/logs/agbd_prithvi_fast-%j.err

set -euo pipefail

echo "=========================================="
echo "AGBD Training: prithvi (fast)"
echo "Job: ${SLURM_JOB_ID} | Started: $(date)"
echo "=========================================="

# Load modules
module load eth_proxy stack/2024-06 

# FAST MODE: Use directories directly (no copying)
echo "FAST MODE: Using directories directly..."
PANGAEA_DIR="/cluster/home/reimannj/pangaea-bench"
VENV_DIR="/cluster/home/reimannj/pangaea-bench-venv"

# Activate environment directly
cd "$PANGAEA_DIR"
source "$VENV_DIR/bin/activate"

# Output directory
OUTDIR="./generated_scripts/results/${SLURM_JOB_ID}_prithvi_fast"
mkdir -p "$OUTDIR"

# Run training (FAST - direct paths like original script)
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
    dataset.root_path=/cluster/work/igp_psr/gsialelli/Data/patches \
    dataset.hdf5_dir=/cluster/work/igp_psr/gsialelli/Data/patches \
    dataset.mapping_path=/cluster/work/igp_psr/gsialelli/Data/AGB \
    dataset.norm_path=/cluster/work/igp_psr/gsialelli/Data/patches \
    seed=42 \
    use_wandb=true \
    work_dir="$OUTDIR"

echo "Training completed: $(date)"
