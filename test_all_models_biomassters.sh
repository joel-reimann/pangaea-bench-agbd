#!/bin/bash
set -euo pipefail

# List of all encoders to test (same as AGBD but excluding the most problematic ones for quick test)
MODELS=(
  satmae_base scalemae prithvi dofa gfmswin remoteclip spectralgpt satlasnet_mi satlasnet_si galileo
  ssl4eo_mae_optical ssl4eo_mae_sar ssl4eo_data2vec ssl4eo_dino ssl4eo_moco
  croma_joint croma_optical croma_sar
  resnet50_pretrained resnet50_scratch vit vit_mi vit_scratch unet_encoder unet_encoder_mi
)

# Quick test subset for faster iteration
# MODELS=(
#   satmae_base scalemae prithvi dofa remoteclip
# )

PANGAEA_HOME="/scratch/final2/pangaea-bench-agbd"
PRETRAINED_DIR="$PANGAEA_HOME/pretrained_models"
GFMSWIN_WEIGHTS="$PRETRAINED_DIR/gfm.pth"

# Check if GFMSwin weights exist (for debug/fix loop)
if [ ! -f "$GFMSWIN_WEIGHTS" ] || [ $(stat -c%s "$GFMSWIN_WEIGHTS" 2>/dev/null || echo 0) -lt 1000000 ]; then
  echo "GFMSwin weights missing or invalid. Please manually download gfm.pth from OneDrive or HuggingFace and place it at: $GFMSWIN_WEIGHTS"
  echo "See README or ask your admin for access."
fi

# BioMassters data path - adjust this to your data location
DATA_PATH="./data/Biomassters"  # Default from config, adjust if needed
LOG_DIR="$PANGAEA_HOME/test_biomassters_logs"
mkdir -p "$LOG_DIR"

# Note: BioMassters uses VV/VH SAR bands (compatible with most models)
# No need for special DOFA patching like with AGBD

# Main loop
for ENCODER in "${MODELS[@]}"; do
  echo "=== Testing BioMassters with model: $ENCODER ==="
  LOGFILE="$LOG_DIR/${ENCODER}_biomassters.log"
  OUTDIR="$LOG_DIR/${ENCODER}_output"
  
  # Note: Using standard regression preprocessing instead of AGBD-specific
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
    dataset.root_path="/tmp" \
    dataset.auto_download=false \
    seed=42 \
    +show_model=false \
    use_wandb=True \
    work_dir="$OUTDIR" \
    task.evaluator.visualization_interval=60 \
    hydra.run.dir="$OUTDIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    > "$LOGFILE" 2>&1 || echo "Model $ENCODER failed, see $LOGFILE"
done

echo "All BioMassters model tests complete. See logs in $LOG_DIR"

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

echo ""
echo "=== Quick Comparison Notes ==="
echo "BioMassters vs AGBD differences:"
echo "- Dataset: BioMassters (ESA biomass) vs AGBD (GEDI biomass)"
echo "- Image size: 256x256 vs 25x25"
echo "- Preprocessing: reg_default vs reg_agbd_original"
echo "- SAR bands: VV/VH vs HH/HV"
echo "- Multi-temporal: 12 months vs single temporal"
echo ""
echo "If predictions are also very low on BioMassters, it's likely just the debug setup (1 epoch, tiny subset)."
echo "If BioMassters gives reasonable predictions but AGBD doesn't, we may have introduced AGBD-specific issues."
