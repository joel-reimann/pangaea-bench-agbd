
#!/bin/bash

set -euo pipefail

declare -a AVAILABLE_MODELS=(
    "prithvi" "scalemae" "remoteclip" "satlasnet_mi" "satlasnet_si" "ssl4eo_mae_sar"
    "croma_optical" "resnet50_pretrained" "resnet50_scratch" "vit" "vit_scratch" "unet_encoder"
)

declare -A EXPERIMENT_CONFIGS=(
    ["debug"]="time=00:30:00 gpus=1 epochs=1"
    ["full"]="time=120:00:00 gpus=8 epochs=20"
)

DEFAULT_ACCOUNT="es_schin"
DEFAULT_EXPERIMENT="full"
DEFAULT_OUTPUT_DIR="/cluster/scratch/$(whoami)/pangaea_experiments"
DEFAULT_PANGAEA_HOME="/cluster/home/$(whoami)/pangaea-bench"
DEFAULT_VENV_HOME="/cluster/home/$(whoami)/pangaea-bench-venv"

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
# SLURM JOB TEMPLATE
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
#SBATCH --tmp=600G
#SBATCH --gpus=${gpus}
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

TMPDIR_PANGAEA="\$TMPDIR/pangaea_workspace"
TMPDIR_VENV="\$TMPDIR/pangaea_venv"
TMPDIR_DATA="\$TMPDIR/agbd_data"
TMPDIR_SPLITS="\$TMPDIR/agbd_splits"

# Create workspace
mkdir -p "\$TMPDIR_DATA" "\$TMPDIR_SPLITS" "\$TMPDIR_PANGAEA" "\$TMPDIR/logs"

echo "Copying codebase..."
rsync -av --exclude '__pycache__' --exclude '.git' --exclude 'wandb' --exclude 'test_agbd_logs' \\
    "${pangaea_home}/" "\$TMPDIR_PANGAEA/"

echo "Copying environment..."
rsync -av "${venv_home}/" "\$TMPDIR_VENV/"

echo "Syncing AGBD data to TMPDIR..."

echo "Syncing dataset..."
rsync --include '*v4_*-20.h5' --exclude '*' -av /cluster/work/igp_psr/gsialelli/Data/patches/ "\$TMPDIR_DATA/"


# Sync split files and metadata
for split in train val test; do
    rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/\${split}_features_2019.csv "\$TMPDIR_DATA/" || echo "Failed to sync \${split} CSV"
done

rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl "\$TMPDIR_DATA/" || echo "Failed to sync statistics"
rsync -av /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl "\$TMPDIR_SPLITS/" || echo "Failed to sync biomes splits"

# Activate environment
cd "\$TMPDIR_PANGAEA"
source "\$TMPDIR_VENV/bin/activate"

export PYTHONPATH="\$TMPDIR_PANGAEA:\${PYTHONPATH:-}"

OUTDIR="${output_dir}/results/\${SLURM_JOB_ID}_${encoder}_${experiment_type}"
mkdir -p "\$OUTDIR"

echo "Starting training"

torchrun --nnodes=1 --nproc_per_node=${gpus} --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \\
    pangaea/run.py \\
    --config-name=train \\
    task=regression \\
    dataset=agbd \\
    encoder=${encoder} \\
    decoder=reg_upernet \\
    criterion=center_pixel_mse \\
    preprocessing=agbd_resize \\
    task.trainer.n_epochs=${epochs} \\
    task.trainer._target_=pangaea.engine.agbd_trainer.AGBDTrainer \\
    task.evaluator._target_=pangaea.engine.agbd_evaluator.AGBDEvaluator \\
    dataset.debug=false \\
    dataset.root_path=/cluster/work/igp_psr/gsialelli/Data/patches \\
    dataset.hdf5_dir=/cluster/work/igp_psr/gsialelli/Data/patches \\
    dataset.mapping_path=/cluster/work/igp_psr/gsialelli/Data/AGB \\
    dataset.norm_path=/cluster/work/igp_psr/gsialelli/Data/patches \\
    use_wandb=true \\
    work_dir="\$OUTDIR"


echo "Training completed: \$(date)"

EOF

    echo "$script_file"
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
    local account="es_schin"

    log_info "Generating SLURM script: $encoder ($experiment_type)"

    local script_file=$(generate_slurm_job "$encoder" "$experiment_type" "$account" "$output_dir" "$pangaea_home" "$venv_home")


    echo ""
    echo "Summary:"
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
    esac
}

main "$@"
