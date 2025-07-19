#!/bin/bash
set -euo pipefail

# Quick test with just a few representative models for faster comparison
MODELS=(
  satmae_base scalemae prithvi remoteclip
)

PANGAEA_HOME="/scratch/final2/pangaea-bench-agbd"
DATA_PATH="./data/Biomassters"  # Default from config, adjust if needed
LOG_DIR="$PANGAEA_HOME/test_biomassters_quick_logs"
mkdir -p "$LOG_DIR"

echo "=== Quick BioMassters Test with 4 Representative Models ==="
echo "This will help us quickly compare if low predictions are dataset-specific or debug-setup related"
echo ""

# Main loop
for ENCODER in "${MODELS[@]}"; do
  echo "=== Testing BioMassters with model: $ENCODER ==="
  LOGFILE="$LOG_DIR/${ENCODER}_biomassters_quick.log"
  OUTDIR="$LOG_DIR/${ENCODER}_output"
  
  torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "$PANGAEA_HOME/pangaea/run.py" \
    --config-name=train \
    dataset=biomassters \
    encoder=$ENCODER \
    decoder=reg_upernet \
    preprocessing=reg_default \
    dataset.debug=True \
    criterion=mse \
    task=regression \
    optimizer=adamw \
    task.trainer.n_epochs=1 \
    task.evaluator.inference_mode=whole \
    dataset.img_size=256 \
    task.trainer.ckpt_interval=1 \
    task.trainer.eval_interval=1 \
    dataset.root_path="$DATA_PATH" \
    seed=42 \
    +show_model=false \
    use_wandb=True \
    work_dir="$OUTDIR" \
    task.evaluator.visualization_interval=60 \
    hydra.run.dir="$OUTDIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    > "$LOGFILE" 2>&1 || echo "Model $ENCODER failed, see $LOGFILE"
done

echo ""
echo "=== Quick Results Summary ==="
echo "Check these key metrics in the logs:"
for ENCODER in "${MODELS[@]}"; do
  LOGFILE="$LOG_DIR/${ENCODER}_biomassters_quick.log"
  if [ -f "$LOGFILE" ]; then
    echo ""
    echo "--- $ENCODER ---"
    # Extract key metrics from log
    echo "Val metrics:"
    grep -E "(val_.*mean_over_test_batches|test_.*mean_over_test_batches)" "$LOGFILE" | tail -5 || echo "No metrics found"
    echo "Prediction samples:"
    grep -E "(predicted_biomass_center|ground_truth_biomass_center)" "$LOGFILE" | tail -3 || echo "No prediction samples found"
  else
    echo "--- $ENCODER --- FAILED (no log file)"
  fi
done

echo ""
echo "=== Comparison Instructions ==="
echo "1. Compare prediction ranges between AGBD and BioMassters"
echo "2. If both datasets show very low predictions: likely debug setup (1 epoch, tiny data)"
echo "3. If only AGBD shows low predictions: we may have AGBD-specific issues"
echo "4. Check WandB visualizations to see if they look reasonable for both datasets"
