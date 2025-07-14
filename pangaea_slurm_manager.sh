#!/bin/bash

# PANGAEA-bench SLURM Job Management System v2.1 (AGBD Integration Complete)
# 
# AGBD INTEGRATION UPDATES (v2.1):
# ================================
# ✅ Fixed preprocessing: reg_agbd_original (preserves original AGBD normalization)
# ✅ Fixed optimizer: adamw (matches AGBD paper recommendations)  
# ✅ Adaptive visualization intervals: debug=60, production=5000 (prevents WandB overflow)
# ✅ All AGBD-specific configs: padding strategy, central pixel scaling, multi-GPU fixes
# ✅ Updated for 300GB full dataset with reasonable visualization frequency
# ✅ Enhanced visualization with SAR/optical panels, fallback grayscale, no yellow boxes
# 
# VISUALIZATION INTERVALS:
# - debug: 60 (every minute, for testing)
# - quick: 500 (every ~8 minutes)  
# - standard: 1000 (every ~16 minutes)
# - thorough: 2000 (every ~33 minutes)
# - production: 5000 (every ~83 minutes)

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

declare -a AVAILABLE_MODELS=(
    "satmae_base" "scalemae" "prithvi" "dofa" "gfmswin" "remoteclip"
    "spectralgpt" "satlasnet_mi" "satlasnet_si" "galileo"
    "ssl4eo_mae_optical" "ssl4eo_mae_sar" "ssl4eo_data2vec" "ssl4eo_dino" "ssl4eo_moco"
    "croma_joint" "croma_optical" "croma_sar"
    "resnet50_pretrained" "resnet50_scratch"
    "vit" "vit_mi" "vit_scratch" "unet_encoder" "unet_encoder_mi"
)
declare -A EXPERIMENT_CONFIGS=(
    ["quick"]="time=30:00:00 epochs=3 seeds=1 lr=0.01 vis_interval=500"
    ["standard"]="time=60:00:00 epochs=5 seeds=1 lr=0.01 vis_interval=1000"
    ["thorough"]="time=120:00:00 epochs=10 seeds=3 lr=0.01 vis_interval=2000"
    ["production"]="time=240:00:00 epochs=20 seeds=5 lr=0.01 vis_interval=5000"
    ["debug"]="time=15:00:00 epochs=1 seeds=1 lr=0.01 vis_interval=60"
)
DEFAULT_ACCOUNT="es_schin"
DEFAULT_EXPERIMENT="standard"
DEFAULT_OUTPUT_DIR="/cluster/scratch/$(whoami)/pangaea_experiments"
DEFAULT_PANGAEA_HOME="/cluster/home/$(whoami)/pangaea-bench"
DEFAULT_VENV_HOME="/cluster/home/$(whoami)/pangaea-bench-venv"

# ============================================================================
# UTILITY LOGGING
# ============================================================================

log_info()     { echo "[INFO] $1"; }
log_success()  { echo "[SUCCESS] $1"; }
log_warning()  { echo "[WARNING] $1"; }
log_error()    { echo "[ERROR] $1"; }

validate_model() {
    local model=$1
    for available_model in "${AVAILABLE_MODELS[@]}"; do
        [[ "$model" == "$available_model" ]] && return 0
    done
    return 1
}

# ============================================================================
# SLURM TEMPLATE GENERATION
# ============================================================================

generate_slurm_template() {
    local job_name=$1 account=$2 time_limit=$3 output_dir=$4 pangaea_home=$5 venv_home=$6
    local encoder=$7 epochs=$8 seed=$9 learning_rate=${10} experiment_type=${11}
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local vis_interval=$(echo $config | grep -o 'vis_interval=[^ ]*' | cut -d'=' -f2)
    # Default visualization interval if not specified in config
    if [ -z "$vis_interval" ]; then
        vis_interval=1000  # Conservative default for production
    fi
    # Allow user override via environment variable
    if [ ! -z "${VIS_INTERVAL:-}" ]; then
        vis_interval=$VIS_INTERVAL
    fi
    cat << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${account}
#SBATCH --time=${time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=800G
#SBATCH --gpus=rtx_4090:8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${output_dir}/logs/slurm-%j.out
#SBATCH --error=${output_dir}/logs/slurm-%j.err

set -euo pipefail
module load eth_proxy stack/2024-06 openblas/0.3.24

mkdir -p "\$TMPDIR/agbd_data" "\$TMPDIR/agbd_splits" "\$TMPDIR/workspace" "\$TMPDIR/logs"
rsync --include '*v4_*-20.h5' --exclude '*' -av /cluster/work/igp_psr/gsialelli/Data/patches/ "\$TMPDIR/agbd_data/"
for split in train val test; do
    rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/\${split}_features_2019.csv "\$TMPDIR/agbd_data/"
done
rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl "\$TMPDIR/agbd_data/"
rsync -av /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl "\$TMPDIR/agbd_splits/"

PANGAEA_HOME="${pangaea_home}"
VENV_HOME="${venv_home}"
cd "\$PANGAEA_HOME"
source "\$VENV_HOME/bin/activate"

PERSISTENT_OUTPUT_DIR="${output_dir}/results/\${SLURM_JOB_ID}_${encoder}_seed${seed}_\$(date +%Y%m%d_%H%M%S)_${experiment_type}"
mkdir -p "\$PERSISTENT_OUTPUT_DIR"

TRAIN_CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 pangaea/run.py \\
    --config-name=train_agbd dataset=agbd encoder=${encoder} decoder=reg_upernet \\
    preprocessing=reg_agbd_original dataset.debug=False criterion=mse task=regression \\
    optimizer=adamw lr_scheduler=step_agbd task.trainer.n_epochs=${epochs} \\
    task.evaluator.inference_mode=whole dataset.img_size=25 task.trainer.ckpt_interval=1 \\
    task.trainer.eval_interval=1 dataset.root_path=\$TMPDIR/agbd_data dataset.hdf5_dir=\$TMPDIR/agbd_data \\
    dataset.mapping_path=\$TMPDIR/agbd_splits dataset.norm_path=\$TMPDIR/agbd_data seed=${seed} \\
    image_processing_strategy=padding use_padding_strategy=true central_pixel_scaling_enabled=true \\
    use_wandb=true work_dir=\$PERSISTENT_OUTPUT_DIR \\
    task.evaluator.visualization_interval=\$vis_interval \\
    hydra.run.dir=\$PERSISTENT_OUTPUT_DIR/hydra_outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
echo "\$TRAIN_CMD" > "\$PERSISTENT_OUTPUT_DIR/command.txt"
eval "\$TRAIN_CMD" > "\$PERSISTENT_OUTPUT_DIR/training.log" 2>&1

echo "\$(date)" > "\$PERSISTENT_OUTPUT_DIR/end_time.txt"
EOF
}

# ============================================================================
# JOB GENERATION
# ============================================================================

generate_single_job() {
    local encoder=$1 experiment_type=$2 seed=$3 account=$4 output_dir=$5 pangaea_home=$6 venv_home=$7
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local time_limit=$(echo $config | grep -o 'time=[^ ]*' | cut -d'=' -f2)
    local epochs=$(echo $config | grep -o 'epochs=[^ ]*' | cut -d'=' -f2)
    local learning_rate=$(echo $config | grep -o 'lr=[^ ]*' | cut -d'=' -f2)
    local job_name="PANGAEA_${encoder}_${experiment_type}_s${seed}"
    local script_file="${output_dir}/jobs/${job_name}.slurm"
    mkdir -p "$(dirname "$script_file")" "${output_dir}/logs"
    generate_slurm_template "$job_name" "$account" "$time_limit" "$output_dir" "$pangaea_home" "$venv_home" "$encoder" "$epochs" "$seed" "$learning_rate" "$experiment_type" > "$script_file"
    echo "$script_file"
}

generate_batch_jobs() {
    local models_string=$1 experiment_type=$2 account=$3 output_dir=$4 pangaea_home=$5 venv_home=$6
    IFS=',' read -ra MODELS <<< "$models_string"
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local num_seeds=$(echo $config | grep -o 'seeds=[^ ]*' | cut -d'=' -f2)
    local job_files=()
    local total_jobs=0
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)
        validate_model "$model" || { log_warning "Skipping invalid model: $model"; continue; }
        for ((seed=1; seed<=num_seeds; seed++)); do
            job_file=$(generate_single_job "$model" "$experiment_type" "$seed" "$account" "$output_dir" "$pangaea_home" "$venv_home")
            job_files+=("$job_file")
            total_jobs=$((total_jobs + 1))
            echo "Generated: $(basename "$job_file")"
        done
    done
    log_success "Generated $total_jobs job files"
    # Submission script
    local submit_script="${output_dir}/submit_all_jobs.sh"
    echo "#!/bin/bash" > "$submit_script"
    echo "set -e" >> "$submit_script"
    for job_file in "${job_files[@]}"; do echo "sbatch $job_file"; done >> "$submit_script"
    chmod +x "$submit_script"
    echo "job_files=(${job_files[*]})"
    echo "submit_script=$submit_script"
    echo "total_jobs=$total_jobs"
}

# ============================================================================
# MODEL GROUPING
# ============================================================================

get_foundation_models() { echo "satmae_base,scalemae,prithvi,dofa,gfmswin,remoteclip,spectralgpt,satlasnet_mi,satlasnet_si,galileo"; }
get_ssl4eo_models()    { echo "ssl4eo_mae_optical,ssl4eo_mae_sar,ssl4eo_data2vec,ssl4eo_dino,ssl4eo_moco"; }
get_croma_models()     { echo "croma_joint,croma_optical,croma_sar"; }
get_baseline_models()  { echo "resnet50_pretrained,resnet50_scratch,vit,vit_mi,vit_scratch,unet_encoder,unet_encoder_mi"; }
get_all_models()       { echo "$(get_foundation_models),$(get_ssl4eo_models),$(get_croma_models),$(get_baseline_models)"; }

# ============================================================================
# HELP AND MODEL LISTING
# ============================================================================

show_help() {
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  generate      Generate SLURM job files"
    echo "  submit        Generate and submit jobs"
    echo "  status        Show job status and recent results"
    echo "  collect       Collect and summarize results"
    echo "  list-models   List all available models"
    echo "  help          Show this help message"
    echo ""
    echo "OPTIONS:"
    echo "  -m, --models MODEL1,MODEL2,...    Models to run (comma-separated)"
    echo "  -a, --all-models                  Run all available models"
    echo "  -f, --foundation-models           Run foundation models only"
    echo "  -s, --ssl4eo-models               Run SSL4EO models only"
    echo "  -c, --croma-models                Run CROMA models only"
    echo "  -b, --baseline-models             Run baseline models only"
    echo "  -e, --experiment TYPE             Experiment type: quick|standard|thorough|production|debug"
    echo "  --vis-interval N                  Override visualization interval (auto: debug=60, production=5000)"
    echo "  -o, --output-dir DIR              Output directory"
    echo "  -p, --pangaea-home DIR            PANGAEA-bench home"
    echo "  -v, --venv-home DIR               Virtual environment"
    echo "  --account ACCOUNT                 SLURM account"
    echo ""
    echo "EXPERIMENT TYPES & VISUALIZATION INTERVALS:"
    echo "  debug:      1 epoch,  vis every 60 batches   (testing only)"
    echo "  quick:      3 epochs, vis every 500 batches  (~8 min)"
    echo "  standard:   5 epochs, vis every 1000 batches (~16 min)"
    echo "  thorough:   10 epochs, vis every 2000 batches (~33 min)"
    echo "  production: 20 epochs, vis every 5000 batches (~83 min)"
    echo ""
}

list_models() {
    echo ""
    echo "Available Models in PANGAEA-bench:"
    echo ""
    echo "Foundation Models: $(get_foundation_models)"
    echo "SSL4EO Models: $(get_ssl4eo_models)"
    echo "CROMA Models: $(get_croma_models)"
    echo "Baseline Models: $(get_baseline_models)"
    echo ""
    echo "Total: ${#AVAILABLE_MODELS[@]} models"
}

# ============================================================================
# STATUS AND COLLECT (REPORTING)
# ============================================================================

show_job_status() {
    local output_dir=$1
    echo ""
    echo "PANGAEA-bench Job Status"
    echo "========================"
    echo ""
    echo "Current Jobs:"
    squeue -u $(whoami) --format="%.10i %.25j %.8T %.10M %.15P %.6D %R" 2>/dev/null || echo "No jobs in queue"
    echo ""
    echo "Recent Results:"
    if [ -d "$output_dir/results" ]; then
        local count=0
        for result_dir in "$output_dir/results"/*; do
            if [ -d "$result_dir" ]; then
                local dirname=$(basename "$result_dir")
                echo "  $dirname"
                count=$((count + 1))
            fi
        done
        if [ $count -eq 0 ]; then
            echo "  No results found"
        fi
    else
        echo "  Results directory not found: $output_dir/results"
    fi
}

collect_results() {
    local output_dir=$1
    local summary_file="${output_dir}/results_summary.csv"
    if [ ! -d "$output_dir/results" ]; then
        log_error "Results directory not found: $output_dir/results"
        return 1
    fi
    log_info "Collecting results from: $output_dir/results"
    echo "job_id,encoder,seed,epochs,learning_rate,experiment_type,status,training_time_seconds,training_time_human,start_time,end_time,exit_code,result_directory" > "$summary_file"
    local total_count=0
    local n_success=0
    local n_failed=0
    for result_dir in "$output_dir/results"/*; do
        if [ -d "$result_dir" ]; then
            local dirname=$(basename "$result_dir")
            local job_id=$(echo "$dirname" | cut -d'_' -f1)
            local encoder=""
            local seed=""
            local epochs=""
            local learning_rate=""
            local experiment_type=""
            local status="unknown"
            local training_time=""
            local training_time_human=""
            local start_time=""
            local end_time=""
            local exit_code=""
            [ -f "$result_dir/encoder.txt" ] && encoder=$(cat "$result_dir/encoder.txt")
            [ -f "$result_dir/seed.txt" ] && seed=$(cat "$result_dir/seed.txt")
            [ -f "$result_dir/epochs.txt" ] && epochs=$(cat "$result_dir/epochs.txt")
            [ -f "$result_dir/learning_rate.txt" ] && learning_rate=$(cat "$result_dir/learning_rate.txt")
            [ -f "$result_dir/experiment_type.txt" ] && experiment_type=$(cat "$result_dir/experiment_type.txt")
            [ -f "$result_dir/status.txt" ] && status=$(cat "$result_dir/status.txt")
            [ -f "$result_dir/training_time_seconds.txt" ] && training_time=$(cat "$result_dir/training_time_seconds.txt")
            [ -f "$result_dir/start_time.txt" ] && start_time=$(cat "$result_dir/start_time.txt")
            [ -f "$result_dir/end_time.txt" ] && end_time=$(cat "$result_dir/end_time.txt")
            [ -f "$result_dir/exit_code.txt" ] && exit_code=$(cat "$result_dir/exit_code.txt")
            if [ -n "$training_time" ] && [ "$training_time" -gt 0 ]; then
                training_time_human="$(($training_time / 3600))h$(($training_time % 3600 / 60))m$(($training_time % 60))s"
            fi
            echo "$job_id,$encoder,$seed,$epochs,$learning_rate,$experiment_type,$status,$training_time,$training_time_human,$start_time,$end_time,$exit_code,$result_dir" >> "$summary_file"
            total_count=$((total_count + 1))
            if [ "$status" = "success" ]; then n_success=$((n_success+1)); fi
            if [ "$status" = "failed" ]; then n_failed=$((n_failed+1)); fi
        fi
    done
    log_success "Results summary saved to: $summary_file"
    # Also create plain text report for test compatibility
    local report_file="${output_dir}/results_report.txt"
    {
        echo "PANGAEA-bench Results Report"
        echo "==========================="
        echo "Total jobs: $total_count"
        echo "Successful: $n_success"
        echo "Failed:     $n_failed"
        echo "See $summary_file for details."
    } > "$report_file"
    log_success "Results report saved to: $report_file"
}

# ============================================================================
# MAIN INTERFACE LOGIC
# ============================================================================

main() {
    local command=""
    local models=""
    local experiment_type="$DEFAULT_EXPERIMENT"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local pangaea_home="$DEFAULT_PANGAEA_HOME"
    local venv_home="$DEFAULT_VENV_HOME"
    local account="$DEFAULT_ACCOUNT"
    local vis_interval=60
    local ckpt_path=""
    # Argument parsing
    while [[ $# -gt 0 ]]; do
        case $1 in
            generate|submit|status|collect|list-models|help|--help|eval|finetune)
                command="$1"
                shift
                ;;
            -m|--models)
                models="$2"; shift 2;;
            -a|--all-models)
                models="$(get_all_models)"; shift;;
            -f|--foundation-models)
                models="$(get_foundation_models)"; shift;;
            -s|--ssl4eo-models)
                models="$(get_ssl4eo_models)"; shift;;
            -c|--croma-models)
                models="$(get_croma_models)"; shift;;
            -b|--baseline-models)
                models="$(get_baseline_models)"; shift;;
            -e|--experiment)
                experiment_type="$2"; shift 2;;
            -o|--output-dir)
                output_dir="$2"; shift 2;;
            -p|--pangaea-home)
                pangaea_home="$2"; shift 2;;
            -v|--venv-home)
                venv_home="$2"; shift 2;;
            --account)
                account="$2"; shift 2;;
            --vis-interval)
                vis_interval="$2"; export VIS_INTERVAL="$2"; shift 2;;
            --ckpt-path)
                ckpt_path="$2"; shift 2;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    # Default to help
    if [ -z "$command" ]; then command="help"; fi
    # Validate experiment type for applicable commands
    if [[ "$command" =~ ^(generate|submit)$ ]]; then
        if [[ -z "${EXPERIMENT_CONFIGS[$experiment_type]+isset}" ]]; then
            log_error "Invalid experiment type: $experiment_type"
            log_info "Available types: ${!EXPERIMENT_CONFIGS[@]}"
            exit 1
        fi
    fi
    # Execute command
    case $command in
        help|--help) show_help;;
        list-models) list_models;;
        generate|submit)
            if [ -z "$models" ]; then
                log_error "No models specified. Use -m or --all-models/--foundation-models/--ssl4eo-models/--croma-models/--baseline-models"
                exit 1
            fi
            log_info "Command: $command"
            log_info "Models: $models"
            log_info "Experiment: $experiment_type (${EXPERIMENT_CONFIGS[$experiment_type]})"
            log_info "Output directory: $output_dir"
            log_info "PANGAEA home: $pangaea_home"
            log_info "Virtual env: $venv_home"
            log_info "Account: $account"
            mkdir -p "$output_dir"
            log_info "Generating batch jobs..."
            generate_batch_jobs "$models" "$experiment_type" "$account" "$output_dir" "$pangaea_home" "$venv_home" > /tmp/batch_output.txt
            submit_script=$(grep "submit_script=" /tmp/batch_output.txt | cut -d'=' -f2)
            total_jobs=$(grep "total_jobs=" /tmp/batch_output.txt | cut -d'=' -f2)
            if [ "$command" = "submit" ]; then
                log_info "Generated $total_jobs job files. Submitting..."
                bash "$submit_script"
                log_success "Batch submission completed!"
            else
                log_success "Generated $total_jobs job files."
                echo ""
                echo "To submit all jobs: bash $submit_script"
                echo "Or submit manually: cd $output_dir/jobs && sbatch *.slurm"
            fi
            ;;
        status) show_job_status "$output_dir";;
        collect) collect_results "$output_dir";;
        eval)
            if [ -z "$models" ] || [ -z "$ckpt_path" ]; then
                log_error "Specify --models and --ckpt-path for evaluation."
                exit 1
            fi
            for model in $(echo $models | tr ',' ' '); do
                generate_eval_job "$model" "$ckpt_path" "$account" "$output_dir" "$pangaea_home" "$venv_home"
            done
            ;;
        finetune)
            if [ -z "$models" ] || [ -z "$ckpt_path" ]; then
                log_error "Specify --models and --ckpt-path for finetuning."
                exit 1
            fi
            for model in $(echo $models | tr ',' ' '); do
                generate_finetune_job "$model" "$ckpt_path" "$account" "$output_dir" "$pangaea_home" "$venv_home"
            done
            ;;
        *) log_error "Unknown command: $command"; show_help; exit 1;;
    esac
}

generate_eval_job() {
    local encoder=$1 ckpt_path=$2 account=$3 output_dir=$4 pangaea_home=$5 venv_home=$6
    local job_name="EVAL_${encoder}_$(date +%Y%m%d_%H%M%S)"
    local script_file="${output_dir}/jobs/${job_name}.slurm"
    mkdir -p "$(dirname "$script_file")" "${output_dir}/logs"
    cat << EOF > "$script_file"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${account}
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=200G
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${output_dir}/logs/slurm-%j.out
#SBATCH --error=${output_dir}/logs/slurm-%j.err

set -euo pipefail
module load eth_proxy stack/2024-06 openblas/0.3.24

PANGAEA_HOME="${pangaea_home}"
VENV_HOME="${venv_home}"
cd "$PANGAEA_HOME"
source "$VENV_HOME/bin/activate"

EVAL_OUTPUT_DIR="${output_dir}/eval_[4m${encoder}[0m_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUTPUT_DIR"

EVAL_CMD="python pangaea/run.py --config-name=train_agbd train=false use_final_ckpt=true ckpt_dir=${ckpt_path} work_dir=$EVAL_OUTPUT_DIR"
echo "$EVAL_CMD" > "$EVAL_OUTPUT_DIR/command.txt"
eval "$EVAL_CMD" > "$EVAL_OUTPUT_DIR/eval.log" 2>&1
EOF
    echo "$script_file"
}

generate_finetune_job() {
    local encoder=$1 ckpt_path=$2 account=$3 output_dir=$4 pangaea_home=$5 venv_home=$6
    local job_name="FINETUNE_${encoder}_$(date +%Y%m%d_%H%M%S)"
    local script_file="${output_dir}/jobs/${job_name}.slurm"
    mkdir -p "$(dirname "$script_file")" "${output_dir}/logs"
    cat << EOF > "$script_file"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${account}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=200G
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${output_dir}/logs/slurm-%j.out
#SBATCH --error=${output_dir}/logs/slurm-%j.err

set -euo pipefail
module load eth_proxy stack/2024-06 openblas/0.3.24

PANGAEA_HOME="${pangaea_home}"
VENV_HOME="${venv_home}"
cd "$PANGAEA_HOME"
source "$VENV_HOME/bin/activate"

FT_OUTPUT_DIR="${output_dir}/finetune_[4m${encoder}[0m_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FT_OUTPUT_DIR"

FT_CMD="python pangaea/run.py --config-name=train_agbd train=true finetune=true ckpt_dir=${ckpt_path} work_dir=$FT_OUTPUT_DIR"
echo "$FT_CMD" > "$FT_OUTPUT_DIR/command.txt"
eval "$FT_CMD" > "$FT_OUTPUT_DIR/finetune.log" 2>&1
EOF
    echo "$script_file"
}
