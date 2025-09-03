set -euo pipefail

CLUSTER_USER="reimannj"
CLUSTER_HOST="euler.ethz.ch" 
CLUSTER_PATH="/cluster/home/reimannj/pangaea-bench"
LOCAL_PATH="/scratch/terminal2/pangaea-bench"

echo "=============================================================="
echo "Local:   $LOCAL_PATH"
echo "Remote:  $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH"
echo ""

# Check space before sync (no deletions)
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
    --exclude='20250715_*' \
    --exclude='*.egg-info/' \
    --delete \
    "$LOCAL_PATH/" \
    "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/"
