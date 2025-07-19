#!/bin/bash
set -euo pipefail

# Quick test with HLS Burn Scars (classification dataset) to see if our pipeline changes work
# We'll treat it as regression just to test the pipeline

MODELS=(
  satmae_base
)

PANGAEA_HOME="/scratch/final2/pangaea-bench-agbd"
DATA_PATH="/scratch/reimannj6/pangaea-bench/data/hlsburnscars"
LOG_DIR="$PANGAEA_HOME/test_hlsburnscars_logs"
mkdir -p "$LOG_DIR"

echo "Testing HLS Burn Scars with regression pipeline (for debugging)"

for ENCODER in "${MODELS[@]}"; do
  echo "=== Testing model: $ENCODER with HLS Burn Scars ==="
  LOGFILE="$LOG_DIR/${ENCODER}_hlsburnscars.log"
  OUTDIR="$LOG_DIR/${ENCODER}_output"
  
  torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "$PANGAEA_HOME/pangaea/run.py" \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=$ENCODER \
    decoder=upernet \
    preprocessing=cls_resize \
    criterion=cross_entropy \
    task=segmentation \
    optimizer=adamw \
    task.trainer.n_epochs=1 \
    task.evaluator.inference_mode=whole \
    dataset.img_size=512 \
    task.trainer.ckpt_interval=1 \
    task.trainer.eval_interval=1 \
    dataset.root_path="$DATA_PATH" \
    dataset.auto_download=false \
    seed=42 \
    +show_model=false \
    use_wandb=True \
    work_dir="$OUTDIR" \
    task.evaluator.visualization_interval=60 \
    hydra.run.dir="$OUTDIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    > "$LOGFILE" 2>&1 || echo "Model $ENCODER failed, see $LOGFILE"
done

echo "HLS Burn Scars test complete. See logs in $LOG_DIR"
