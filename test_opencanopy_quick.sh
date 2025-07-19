#!/bin/bash
set -euo pipefail

# Quick test with OpenCanopy (canopy height regression) to compare with AGBD
MODELS=(
  satmae_base
  # scalemae prithvi dofa remoteclip
)

PANGAEA_HOME="/scratch/final2/pangaea-bench-agbd"
DATA_PATH="./data/canopy_height"  # OpenCanopy auto-downloads here
LOG_DIR="$PANGAEA_HOME/test_opencanopy_logs"
mkdir -p "$LOG_DIR"

echo "=== Testing OpenCanopy Canopy Height Regression ==="
echo "This will test if low predictions are due to debug setup or AGBD-specific changes"

# Main loop
for ENCODER in "${MODELS[@]}"; do
  echo "=== Testing model: $ENCODER ==="
  LOGFILE="$LOG_DIR/${ENCODER}_opencanopy.log"
  OUTDIR="$LOG_DIR/${ENCODER}_output"
  torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "$PANGAEA_HOME/pangaea/run.py" \
    --config-name=train \
    dataset=opencanopy \
    encoder=$ENCODER \
    decoder=reg_upernet \
    preprocessing=reg_default \
    dataset.debug=True \
    criterion=mse \
    task=regression \
    optimizer=adamw \
    task.trainer.n_epochs=1 \
    task.evaluator.inference_mode=whole \
    dataset.img_size=224 \
    task.trainer.ckpt_interval=1 \
    task.trainer.eval_interval=1 \
    dataset.root_path="$DATA_PATH" \
    seed=42 \
    use_wandb=True \
    work_dir="$OUTDIR" \
    +show_model=True \
    task.evaluator.visualization_interval=60 \
    hydra.run.dir="$OUTDIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    > "$LOGFILE" 2>&1 || echo "Model $ENCODER failed, see $LOGFILE"
done

echo "OpenCanopy test complete. See logs in $LOG_DIR"
echo ""
echo "COMPARISON ANALYSIS:"
echo "- If OpenCanopy also shows very low predictions -> debug setup issue (1 epoch, small data)"
echo "- If OpenCanopy shows reasonable predictions -> AGBD-specific issue"
echo ""
echo "Check WandB for visualization comparisons between AGBD and OpenCanopy"
