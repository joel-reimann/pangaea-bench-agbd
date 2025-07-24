#!/bin/bash

# PANGAEA-bench AGBD SLURM Manager (Clean Version)
# Simple SLURM wrapper matching test_all_models_agbd.sh

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

declare -a AVAILABLE_MODELS=(
    "prithvi" "satmae_base" "scalemae" "dofa" "gfmswin" "remoteclip"
    "spectralgpt" "satlasnet_mi" "satlasnet_si" "galileo"
    "ssl4eo_mae_optical" "ssl4eo_mae_sar" "ssl4eo_data2vec" "ssl4eo_dino" "ssl4eo_moco"
    "croma_joint" "croma_optical" "croma_sar"
    "resnet50_pretrained" "resnet50_scratch"
    "vit" "vit_mi" "vit_scratch" "unet_encoder" "unet_encoder_mi"
)

# Simple experiment types
declare -A EXPERIMENT_CONFIGS=(
    ["minimal"]="time=00:30:00 gpus=1 epochs=1"
    ["debug"]="time=02:00:00 gpus=2 epochs=5" 
    ["full"]="time=30:00:00 gpus=8 epochs=20"
    ["fast"]="time=30:00:00 gpus=8 epochs=20"
)

# Defaults
DEFAULT_ACCOUNT="es_schin"
DEFAULT_EXPERIMENT="debug"
DEFAULT_OUTPUT_DIR="/cluster/scratch/$(whoami)/pangaea_experiments"
DEFAULT_PANGAEA_HOME="/cluster/home/$(whoami)/pangaea-bench"
DEFAULT_VENV_HOME="/cluster/home/$(whoami)/pangaea-bench-venv"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() { echo "[INFO $(date '+%H:%M:%S')] $1"; }
log_error() { echo "[ERROR $(date '+%H:%M:%S')] $1"; }

validate_model() {
    local model=$1
    for available_model in "${AVAILABLE_MODELS[@]}"; do
        [[ "$model" == "$available_model" ]] && return 0
    done
    return 1
}

extract_config_value() {
    local config=$1
    local key=$2
    echo "$config" | grep -o "${key}=[^ ]*" | cut -d'=' -f2 || echo ""
}

# ============================================================================
# SLURM JOB TEMPLATE (CLEAN VERSION MATCHING LOCAL TEST)
# ============================================================================

generate_slurm_job() {
    local encoder=$1
    local experiment_type=$2
    local account=$3
    local output_dir=$4
    local pangaea_home=$5
    local venv_home=$6
    
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local time_limit=$(extract_config_value "$config" "time")
    local gpus=$(extract_config_value "$config" "gpus")
    local epochs=$(extract_config_value "$config" "epochs")
    
    local job_name="agbd_${encoder}_${experiment_type}"
    local script_file="${output_dir}/jobs/${job_name}.slurm"
    
    mkdir -p "$(dirname "$script_file")" "${output_dir}/logs"
    
    cat > "$script_file" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${account}
#SBATCH --time=${time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=400G
#SBATCH --gpus=rtx_3090:${gpus}
#SBATCH --output=${output_dir}/logs/${job_name}-%j.out
#SBATCH --error=${output_dir}/logs/${job_name}-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=========================================="
echo "AGBD Training: ${encoder} (${experiment_type})"
echo "Job: \${SLURM_JOB_ID} | Started: \$(date)"
echo "=========================================="

# Load modules
module load eth_proxy stack/2024-06 

EOF

    # Different setup for fast vs normal experiments
    if [[ "$experiment_type" == "fast" ]]; then
        cat >> "$script_file" << EOF
# FAST MODE: Use directories directly (no copying)
echo "FAST MODE: Using directories directly..."
PANGAEA_DIR="${pangaea_home}"
VENV_DIR="${venv_home}"

# Activate environment directly
cd "\$PANGAEA_DIR"
source "\$VENV_DIR/bin/activate"

# Add current directory to Python path for imports
export PYTHONPATH="\$PANGAEA_DIR:\${PYTHONPATH:-}"

# Output directory
OUTDIR="${output_dir}/results/\${SLURM_JOB_ID}_${encoder}_${experiment_type}"
mkdir -p "\$OUTDIR"

# Run training (FAST - direct paths like original script)
echo "Starting training..."
torchrun --nnodes=1 --nproc_per_node=${gpus} --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \\
    pangaea/run.py \\
    --config-name=train_agbd \\
    dataset=agbd \\
    encoder=${encoder} \\
    decoder=reg_upernet \\
    criterion=mse \\
    task=regression \\
    task.trainer.n_epochs=${epochs} \\
    task.evaluator.inference_mode=whole \\
    task.trainer.ckpt_interval=1 \\
    task.trainer.eval_interval=1 \\
    dataset.debug=false \\
    dataset.root_path=/cluster/work/igp_psr/gsialelli/Data/patches \\
    dataset.hdf5_dir=/cluster/work/igp_psr/gsialelli/Data/patches \\
    dataset.mapping_path=/cluster/work/igp_psr/gsialelli/Data/AGB \\
    dataset.norm_path=/cluster/work/igp_psr/gsialelli/Data/patches \\
    preprocessing.train.preprocessor_cfg.2.stats_path=/cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl \\
    preprocessing.val.preprocessor_cfg.2.stats_path=/cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl \\
    preprocessing.test.preprocessor_cfg.2.stats_path=/cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl \\
    seed=42 \\
    use_wandb=true \\
    work_dir="\$OUTDIR"
EOF
    else
        cat >> "$script_file" << EOF
# NORMAL MODE: Copy to TMPDIR for isolation
# Setup workspace (matching original structure)
TMPDIR_PANGAEA="\$TMPDIR/pangaea_workspace"
TMPDIR_VENV="\$TMPDIR/pangaea_venv"
TMPDIR_DATA="\$TMPDIR/agbd_data"
TMPDIR_SPLITS="\$TMPDIR/agbd_splits"

# Create workspace structure (like original)
mkdir -p "\$TMPDIR_DATA" "\$TMPDIR_SPLITS" "\$TMPDIR_PANGAEA" "\$TMPDIR/logs"

# Copy codebase
echo "Copying codebase..."
rsync -av --exclude '__pycache__' --exclude '.git' --exclude 'wandb' --exclude 'test_agbd_logs' \\
    "${pangaea_home}/" "\$TMPDIR_PANGAEA/"

# Copy environment  
echo "Copying environment..."
rsync -av "${venv_home}/" "\$TMPDIR_VENV/"

# Copy AGBD data with selective patterns (matching original exactly)
echo "Syncing AGBD data to TMPDIR..."

if [ "${experiment_type}" = "debug" ] || [ "${experiment_type}" = "minimal" ]; then
    # For debug/minimal: sync v4 data with size limit for speed
    echo "Debug mode: syncing minimal dataset..."
    rsync --include '*v4_*-20.h5' --exclude '*' -av /cluster/work/igp_psr/gsialelli/Data/patches/ "\$TMPDIR_DATA/" --max-size=100M || true
else
    # For production: sync full v4 dataset
    echo "Production mode: syncing full dataset..."
    rsync --include '*v4_*-20.h5' --exclude '*' -av /cluster/work/igp_psr/gsialelli/Data/patches/ "\$TMPDIR_DATA/"
fi

# Sync split files and metadata (exactly like original)
for split in train val test; do
    rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/\${split}_features_2019.csv "\$TMPDIR_DATA/" || echo "Failed to sync \${split} CSV"
done
rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl "\$TMPDIR_DATA/" || echo "Failed to sync statistics"
rsync -av /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl "\$TMPDIR_SPLITS/" || echo "Failed to sync biomes splits"

# Activate environment
cd "\$TMPDIR_PANGAEA"
source "\$TMPDIR_VENV/bin/activate"

# Add current directory to Python path for imports
export PYTHONPATH="\$TMPDIR_PANGAEA:\${PYTHONPATH:-}"

# Output directory
OUTDIR="${output_dir}/results/\${SLURM_JOB_ID}_${encoder}_${experiment_type}"
mkdir -p "\$OUTDIR"

# Run training (EXACT SAME COMMAND AS LOCAL SCRIPT)
echo "Starting training..."
torchrun --nnodes=1 --nproc_per_node=${gpus} --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \\
    pangaea/run.py \\
    --config-name=train_agbd \\
    dataset=agbd \\
    encoder=${encoder} \\
    decoder=reg_upernet \\
    criterion=mse \\
    task=regression \\
    task.trainer.n_epochs=${epochs} \\
    task.evaluator.inference_mode=whole \\
    task.trainer.ckpt_interval=1 \\
    task.trainer.eval_interval=1 \\
    dataset.root_path="\$TMPDIR_DATA" \\
    dataset.hdf5_dir="\$TMPDIR_DATA" \\
    dataset.mapping_path="\$TMPDIR_SPLITS" \\
    dataset.norm_path="\$TMPDIR_DATA" \\
    preprocessing.train.preprocessor_cfg.2.stats_path="\$TMPDIR_DATA/statistics_subset_2019-2020-v4_new.pkl" \\
    preprocessing.val.preprocessor_cfg.2.stats_path="\$TMPDIR_DATA/statistics_subset_2019-2020-v4_new.pkl" \\
    preprocessing.test.preprocessor_cfg.2.stats_path="\$TMPDIR_DATA/statistics_subset_2019-2020-v4_new.pkl" \\
    seed=42 \\
    use_wandb=true \\
    work_dir="\$OUTDIR"
EOF
    fi

    cat >> "$script_file" << EOF

echo "Training completed: \$(date)"
EOF

    echo "$script_file"
}

# ============================================================================
# MAIN INTERFACE
# ============================================================================

show_help() {
    cat << EOF
PANGAEA-bench AGBD SLURM Manager (Clean Version)
==============================================

Simple SLURM wrapper matching test_all_models_agbd.sh

USAGE:
  $0 submit <encoder> [experiment_type] [account] [output_dir] [pangaea_home] [venv_home]
  $0 generate <encoder> [experiment_type] [output_dir] [pangaea_home] [venv_home]
  $0 list-models
  $0 help

EXAMPLES:
  $0 submit prithvi minimal
  $0 generate satmae_base debug                    # Create script in ./generated_scripts/
  $0 generate dofa full ./my_scripts               # Create script in custom directory
  $0 submit dofa full es_schin /cluster/scratch/me/agbd

EXPERIMENT TYPES:
  minimal  - 1 GPU,  1 epoch,  30min  (quick test with copying)
  debug    - 2 GPUs, 5 epochs, 2hrs   (development with copying)
  full     - 4 GPUs, 20 epochs, 8hrs  (production with copying)
  fast     - 1 GPU,  1 epoch,  15min  (super quick, NO copying)

AVAILABLE MODELS:
EOF
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo "  $model"
    done
}

list_models() {
    echo "Available models:"
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo "  $model"
    done
}

submit_job() {
    local encoder=${1:-}
    local experiment_type=${2:-$DEFAULT_EXPERIMENT}
    local account=${3:-$DEFAULT_ACCOUNT}
    local output_dir=${4:-$DEFAULT_OUTPUT_DIR}
    local pangaea_home=${5:-$DEFAULT_PANGAEA_HOME}
    local venv_home=${6:-$DEFAULT_VENV_HOME}
    
    # Validate inputs
    if [[ -z "$encoder" ]]; then
        log_error "Encoder model required"
        echo "Usage: $0 submit <encoder> [experiment_type]"
        exit 1
    fi
    
    if ! validate_model "$encoder"; then
        log_error "Invalid model: $encoder"
        echo "Available models:"
        list_models
        exit 1
    fi
    
    if [[ -z "${EXPERIMENT_CONFIGS[$experiment_type]+isset}" ]]; then
        log_error "Invalid experiment type: $experiment_type"
        echo "Available types: ${!EXPERIMENT_CONFIGS[@]}"
        exit 1
    fi
    
    # Generate and submit job
    log_info "Generating SLURM job for $encoder ($experiment_type)"
    local script_file=$(generate_slurm_job "$encoder" "$experiment_type" "$account" "$output_dir" "$pangaea_home" "$venv_home")
    
    log_info "Submitting job: $script_file"
    sbatch "$script_file"
}

generate_job() {
    local encoder=${1:-}
    local experiment_type=${2:-$DEFAULT_EXPERIMENT}
    local output_dir=${3:-"./generated_scripts"}
    local pangaea_home=${4:-$DEFAULT_PANGAEA_HOME}
    local venv_home=${5:-$DEFAULT_VENV_HOME}
    local account="es_schin"  # Default for generation
    
    # Validate inputs
    if [[ -z "$encoder" ]]; then
        log_error "Encoder model required"
        echo "Usage: $0 generate <encoder> [experiment_type] [output_dir]"
        exit 1
    fi
    
    if ! validate_model "$encoder"; then
        log_error "Invalid model: $encoder"
        echo "Available models:"
        list_models
        exit 1
    fi
    
    if [[ -z "${EXPERIMENT_CONFIGS[$experiment_type]+isset}" ]]; then
        log_error "Invalid experiment type: $experiment_type"
        echo "Available types: ${!EXPERIMENT_CONFIGS[@]}"
        exit 1
    fi
    
    # Generate job script for inspection
    log_info "Generating SLURM script for inspection: $encoder ($experiment_type)"
    local script_file=$(generate_slurm_job "$encoder" "$experiment_type" "$account" "$output_dir" "$pangaea_home" "$venv_home")
    
    log_info "✅ Script generated: $script_file"
    log_info "📄 You can inspect it with: cat $script_file"
    log_info "🚀 To submit it later: sbatch $script_file"
    
    echo ""
    echo "Script summary:"
    echo "  Encoder: $encoder"
    echo "  Type: $experiment_type"
    echo "  GPUs: $(extract_config_value "${EXPERIMENT_CONFIGS[$experiment_type]}" "gpus")"
    echo "  Epochs: $(extract_config_value "${EXPERIMENT_CONFIGS[$experiment_type]}" "epochs")"
    echo "  Time: $(extract_config_value "${EXPERIMENT_CONFIGS[$experiment_type]}" "time")"
}

main() {
    case "${1:-help}" in
        submit)
            shift
            submit_job "$@"
            ;;
        generate)
            shift
            generate_job "$@"
            ;;
        list-models)
            list_models
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
