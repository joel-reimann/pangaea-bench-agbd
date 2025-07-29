#!/bin/bash
set -euo pipefail

# ALIGNMENT FIX TEST: Start with just ViT-based models to validate alignment fix
# MODELS=(
  # satmae_base
  # ssl4eo_mae_optical  # Add more after first test succeeds
  # scalemae
  # prithvi
# )

# Full model list (uncomment after alignment is validated)
MODELS=(
  satmae_base scalemae prithvi dofa gfmswin remoteclip spectralgpt satlasnet_mi satlasnet_si galileo
  ssl4eo_mae_optical ssl4eo_mae_sar ssl4eo_data2vec ssl4eo_dino ssl4eo_moco
  croma_joint croma_optical croma_sar
  resnet50_pretrained resnet50_scratch vit vit_mi vit_scratch unet_encoder unet_encoder_mi
)



# Download GFMSwin weights if missing (for debug/fix loop)
PANGAEA_HOME="/scratch/final3/pangaea-bench-agbd"
PRETRAINED_DIR="$PANGAEA_HOME/pretrained_models"
# GFMSWIN_WEIGHTS="$PRETRAINED_DIR/gfm.pth"
# Check if file exists and is at least 1MB (valid)
# if [ ! -f "$GFMSWIN_WEIGHTS" ] || [ $(stat -c%s "$GFMSWIN_WEIGHTS" 2>/dev/null || echo 0) -lt 1000000 ]; then
#   echo "GFMSwin weights missing or invalid. Please manually download gfm.pth from OneDrive or HuggingFace and place it at: $GFMSWIN_WEIGHTS"
#   echo "See README or ask your admin for access."
# fi

DATA_PATH="/scratch/reimannj/pangaea_agbd_integration_final/data/agbd"
LOG_DIR="$PANGAEA_HOME/test_agbd_logs"
mkdir -p "$LOG_DIR"

# Patch for DOFA: Map AGBD SAR bands to DOFA-expected keys for wave_list
# (DOFA expects 'VV' and 'VH', but AGBD uses 'HH' and 'HV')
# CRITICAL: Using reg_agbd_original preprocessing to match original AGBD normalization
# DOFA_WAVE_PATCH=''
# if [[ " ${MODELS[@]} " =~ "dofa" ]]; then
#   # If running dofa, patch config to map HH->VV and HV->VH for wave_list
#   export DOFA_WAVE_PATCH=1
#   echo "[DOFA PATCH] Mapping AGBD SAR bands HH/HV to DOFA SAR bands VV/VH in wave_list."
# fi

# Main loop
for ENCODER in "${MODELS[@]}"; do
  echo "=== Testing model: $ENCODER ==="
  LOGFILE="$LOG_DIR/${ENCODER}_agbd.log"
  OUTDIR="$LOG_DIR/${ENCODER}_output"
  torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "$PANGAEA_HOME/pangaea/run.py" \
    --config-name=train_agbd \
    dataset=agbd \
    encoder=$ENCODER \
    decoder=reg_upernet \
    dataset.debug=True \
    criterion=mse \
    task=regression \
    task.trainer.n_epochs=3 \
    task.evaluator.inference_mode=whole \
    dataset.img_size=32 \
    task.trainer.ckpt_interval=1 \
    task.trainer.eval_interval=1 \
    dataset.root_path="$DATA_PATH" \
    dataset.hdf5_dir="$DATA_PATH" \
    dataset.mapping_path="$DATA_PATH" \
    dataset.norm_path="$DATA_PATH" \
    seed=42 \
    image_processing_strategy=padding \
    use_padding_strategy=true \
    central_pixel_scaling_enabled=false \
    use_wandb=True \
    work_dir="$OUTDIR" \
    task.evaluator.visualization_interval=60 \
    hydra.run.dir="$OUTDIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    > "$LOGFILE" 2>&1 || echo "Model $ENCODER failed, see $LOGFILE"
done

echo "All model tests complete. See logs in $LOG_DIR"

echo "\n=== Visualization Output Check ==="
MARKER_FILE="visualization_done.txt"
for ENCODER in "${MODELS[@]}"; do
  OUTDIR="$LOG_DIR/${ENCODER}_output"
  if [ -f "$OUTDIR/$MARKER_FILE" ]; then
    echo "[OK] $ENCODER: Visualization marker found ($MARKER_FILE)"
  else
    echo "[WARN] $ENCODER: No visualization marker found in $OUTDIR (wandb logic may not have run)"
  fi
done
