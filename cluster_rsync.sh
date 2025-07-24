#!/bin/bash

# PANGAEA-bench Cluster Rsync Script - SAFE VERSION (No local deletions)
# Syncs only essential files, excludes debug outputs and test data

set -euo pipefail

# Configuration
CLUSTER_USER="reimannj"
CLUSTER_HOST="euler.ethz.ch" 
CLUSTER_PATH="/cluster/home/reimannj/pangaea-bench"
LOCAL_PATH="/scratch/final2/pangaea-bench-agbd"

echo "🚀 PANGAEA-bench Cluster Sync (Safe Mode - No Local Deletions)"
echo "=============================================================="
echo "Local:   $LOCAL_PATH"
echo "Remote:  $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH"
echo ""

# Check space before sync (no deletions)
echo "📊 Current local space usage:"
du -sh "$LOCAL_PATH"
echo ""

echo "📤 Syncing to cluster (excluding large files)..."
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='test_*_logs/' \
    --exclude='wandb/' \
    --exclude='data/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='.DS_Store' \
    --exclude='Thumbs.db' \
    --exclude='.pytest_cache/' \
    --exclude='.coverage' \
    --exclude='htmlcov/' \
    --exclude='2025*' \
    --exclude='*.egg-info/' \
    --delete \
    "$LOCAL_PATH/" \
    "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/"

echo ""
echo "🎉 Sync complete! (No local files were deleted)"
echo ""
echo "📋 What was synced to cluster:"
echo "  ✅ Source code (pangaea/, configs/, etc.)"
echo "  ✅ Pretrained models (~11GB)"
echo "  ✅ Scripts and documentation"
echo "  ✅ Requirements and setup files"
echo ""
echo "❌ What was excluded from sync:"
echo "  ❌ Git history (.git/ - 11GB) - KEPT LOCALLY"
echo "  ❌ Test outputs (test_*_logs/ - 61GB) - KEPT LOCALLY"
echo "  ❌ Downloaded data (data/ - 4.9GB) - KEPT LOCALLY"
echo "  ❌ Local WandB logs (wandb/ - 77MB) - KEPT LOCALLY"
echo "  ❌ Python cache and temp files - KEPT LOCALLY"
echo ""
echo "🎯 Estimated cluster usage: ~15-20GB (well under 50GB limit)"
echo "💾 All files remain safe on your local machine!"
echo ""
echo "Next steps:"
echo "1. SSH to cluster: ssh $CLUSTER_USER@$CLUSTER_HOST"
echo "2. cd pangaea-bench"
echo "3. Run your SLURM manager script!"
