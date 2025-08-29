#!/bin/bash

set -euo pipefail

# Parse command line arguments
N_EPOCHS=${1:-20}
NPROC_PER_NODE=${2:-1}
PANGAEA_HOME=${3:-"/scratch2/reimannj/pangaea-bench"}
DATA_PATH=${4:-"/scratch2/reimannj/AGBD/Models/Data"}
LOG_DIR=${5:-"$PANGAEA_HOME/agbd_logs"}

# Print usage if needed
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [N_EPOCHS] [NPROC_PER_NODE] [PANGAEA_HOME] [DATA_PATH] [LOG_DIR]"
    echo "  N_EPOCHS: Number of training epochs (default: 20)"
    echo "  NPROC_PER_NODE: Number of processes per node (default: 1)"
    echo "  PANGAEA_HOME: Path to pangaea-bench directory (default: /scratch2/reimannj/pangaea-bench)"
    echo "  DATA_PATH: Path to data directory (default: /scratch2/reimannj/AGBD/Models/Data)"
    echo "  LOG_DIR: Path to log directory (default: \$PANGAEA_HOME/agbd_logs)"
    exit 0
fi

# Debugging Flags
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export HYDRA_FULL_ERROR=1
# export TORCH_LOGS="+all"
# export CUDA_LAUNCH_BLOCKING=1

echo "=== Starting model runs ==="
echo "N_EPOCHS: $N_EPOCHS"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "PANGAEA_HOME: $PANGAEA_HOME"
echo "DATA_PATH: $DATA_PATH"
echo "LOG_DIR: $LOG_DIR"

nvidia-smi || echo "WARNING: nvidia-smi not available"

MODELS=(
)

mkdir -p "$LOG_DIR"

for ENCODER in "${MODELS[@]}"; do
  LOGFILE="$LOG_DIR/${ENCODER}_$(date +%Y%m%d_%H%M%S)_agbd.log"
  OUTDIR="$LOG_DIR/${ENCODER}_output"

  if [[ "$ENCODER" == "unet_encoder"* ]]; then
    DECODER="reg_unet"
  else
    DECODER="reg_upernet"
  fi
  
  echo "Starting torchrun for $ENCODER with decoder $DECODER"
  
  torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    "$PANGAEA_HOME/pangaea/run.py" \
    --config-name=train \
    task=regression \
    dataset=agbd \
    encoder=$ENCODER \
    decoder=$DECODER \
    preprocessing=agbd_resize \
    criterion=center_pixel_mse \
    task.trainer._target_=pangaea.engine.agbd_trainer.AGBDTrainer \
    task.evaluator._target_=pangaea.engine.agbd_evaluator.AGBDEvaluator \
    dataset.debug=true \
    task.trainer.n_epochs=$N_EPOCHS \
    dataset.root_path="$DATA_PATH" \
    dataset.hdf5_dir="$DATA_PATH" \
    dataset.mapping_path="$DATA_PATH" \
    dataset.norm_path="$DATA_PATH" \
    use_wandb=true \
    work_dir="$OUTDIR" \
    hydra.run.dir="$OUTDIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    > "$LOGFILE" 2>&1
    
  EXIT_CODE=$?
  echo "torchrun completed with exit code: $EXIT_CODE"
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Model $ENCODER passed"
  else
    echo "✗ Model $ENCODER failed with exit code $EXIT_CODE, see $LOGFILE"
    tail -20 "$LOGFILE" || echo "Could not read log file"
  fi
  
  pkill -f "run.py" || true
  sleep 2
done

echo "=== All model tests complete ==="
ls -la "$LOG_DIR" | head -20
