#!/bin/bash

# PANGAEA-bench AGBD Integration - Cluster Sync Script
# Syncs all AGBD-related changes from /scratch/final2/pangaea-bench-agbd/ to ~/pangaea-bench/

set -euo pipefail

SOURCE_DIR="/scratch/final2/pangaea-bench-agbd"
TARGET_DIR="$HOME/pangaea-bench"

echo "🚀 Syncing AGBD integration files to cluster..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Critical files that MUST be synced
CRITICAL_FILES=(
    "agbd_visualization.py"
    "pangaea/engine/evaluator.py"
    "pangaea/datasets/agbd.py"
    "configs/dataset/agbd.yaml"
    "configs/train_agbd.yaml"
    "configs/preprocessing/reg_agbd_original.yaml"
    "configs/preprocessing/reg_agbd_padding.yaml"
    "configs/optimizer/adamw.yaml"
    "configs/optimizer/adam_agbd.yaml"
    "configs/lr_scheduler/step_agbd.yaml"
)

# Sync critical files
echo "📁 Syncing critical AGBD files..."
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        echo "  ✅ $file"
        mkdir -p "$TARGET_DIR/$(dirname "$file")"
        cp "$SOURCE_DIR/$file" "$TARGET_DIR/$file"
    else
        echo "  ❌ MISSING: $file"
    fi
done

# Sync entire configs directory to be safe
echo "📁 Syncing configs directory..."
rsync -av "$SOURCE_DIR/configs/" "$TARGET_DIR/configs/"

# Sync pangaea directory (selective)
echo "📁 Syncing pangaea core files..."
rsync -av --include="*.py" --exclude="__pycache__" "$SOURCE_DIR/pangaea/" "$TARGET_DIR/pangaea/"

echo "✅ Sync complete!"
echo ""
echo "🔍 Verifying critical files exist on cluster:"
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$TARGET_DIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ MISSING: $file"
    fi
done

echo ""
echo "🎯 Ready to run SLURM jobs with AGBD integration!"
