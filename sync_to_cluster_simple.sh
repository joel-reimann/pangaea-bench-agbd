#!/bin/bash

# PANGAEA-bench AGBD Integration - Simple Full Repo Sync
# Syncs the entire AGBD repo to cluster with smart exclusions

set -euo pipefail

SOURCE_DIR="/scratch/final2/pangaea-bench-agbd"
TARGET_DIR="$HOME/pangaea-bench"

echo "🚀 Syncing entire AGBD repo to cluster..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "📁 Syncing entire repository with exclusions..."
rsync -av \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'test_agbd_logs' \
    --exclude 'wandb' \
    --exclude '.pytest_cache' \
    --exclude '*.log' \
    --exclude 'logs/' \
    --exclude '*.pth' \
    --exclude '20250*' \
    --exclude 'results/' \
    --exclude 'checkpoints/' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

echo "✅ Sync complete!"
echo ""
echo "📊 Synced files summary:"
echo "  Configs: $(find "$TARGET_DIR/configs" -name "*.yaml" 2>/dev/null | wc -l) YAML files"
echo "  Python:  $(find "$TARGET_DIR" -name "*.py" 2>/dev/null | wc -l) Python files"
echo "  Total:   $(find "$TARGET_DIR" -type f 2>/dev/null | wc -l) files"
echo ""
echo "🎯 Ready to run SLURM jobs with complete AGBD integration!"
