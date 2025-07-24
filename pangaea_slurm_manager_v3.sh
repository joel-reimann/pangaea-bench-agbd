#!/bin/bash

# PANGAEA-bench SLURM Job Management System v3.0 (Full Isolation & Debug Mode)
# 
# KEY IMPROVEMENTS IN v3.0:
# ========================
# - Full codebase rsync to TMPDIR for isolation and performance
# - Always copy virtual environment for complete isolation
# - Minimal debug mode: 1 model, 1 GPU, tiny subset, 50 samples, 1 epoch
# - Fine-grained control: epochs, batch size, learning rate, vis interval
# - Robust error handling and recovery
# - Professional structure with DRY principles
# - Enhanced logging and debugging capabilities
# - Adaptive resource allocation based on experiment type

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

# Experiment configurations with resource allocation
declare -A EXPERIMENT_CONFIGS=(
    ["minimal"]="time=10:00:00 epochs=1 seeds=1 lr=0.01 vis_interval=20 gpus=1 debug=true max_samples=50"
    ["debug"]="time=15:00:00 epochs=1 seeds=1 lr=0.01 vis_interval=60 gpus=2 debug=true max_samples=500"
    ["quick"]="time=30:00:00 epochs=3 seeds=1 lr=0.01 vis_interval=500 gpus=4 debug=false"
    ["standard"]="time=60:00:00 epochs=5 seeds=1 lr=0.01 vis_interval=1000 gpus=8 debug=false"
    ["thorough"]="time=120:00:00 epochs=10 seeds=3 lr=0.01 vis_interval=2000 gpus=8 debug=false"
    ["production"]="time=240:00:00 epochs=20 seeds=5 lr=0.01 vis_interval=5000 gpus=8 debug=false"
)

# Default configuration
DEFAULT_ACCOUNT="es_schin"
DEFAULT_EXPERIMENT="standard"
DEFAULT_OUTPUT_DIR="/cluster/scratch/$(whoami)/pangaea_experiments"
DEFAULT_PANGAEA_HOME="/cluster/home/$(whoami)/pangaea-bench"
DEFAULT_VENV_HOME="/cluster/home/$(whoami)/pangaea-bench-venv"
DEFAULT_BATCH_SIZE="32"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info()     { echo "[INFO $(date '+%H:%M:%S')] $1"; }
log_success()  { echo "[SUCCESS $(date '+%H:%M:%S')] $1"; }
log_warning()  { echo "[WARNING $(date '+%H:%M:%S')] $1"; }
log_error()    { echo "[ERROR $(date '+%H:%M:%S')] $1"; }
log_debug()    { echo "[DEBUG $(date '+%H:%M:%S')] $1"; }

validate_model() {
    local model=$1
    for available_model in "${AVAILABLE_MODELS[@]}"; do
        [[ "$model" == "$available_model" ]] && return 0
    done
    return 1
}

validate_experiment_type() {
    local exp_type=$1
    [[ -n "${EXPERIMENT_CONFIGS[$exp_type]+isset}" ]]
}

extract_config_value() {
    local config=$1
    local key=$2
    echo "$config" | grep -o "${key}=[^ ]*" | cut -d'=' -f2 || echo ""
}

# ============================================================================
# ADVANCED SLURM TEMPLATE GENERATION
# ============================================================================

generate_slurm_template() {
    local job_name=$1
    local account=$2
    local time_limit=$3
    local output_dir=$4
    local pangaea_home=$5
    local venv_home=$6
    local encoder=$7
    local epochs=$8
    local seed=$9
    local learning_rate=${10}
    local experiment_type=${11}
    local batch_size=${12}
    
    # Extract configuration
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local vis_interval=$(extract_config_value "$config" "vis_interval")
    local gpus=$(extract_config_value "$config" "gpus")
    local debug_mode=$(extract_config_value "$config" "debug")
    local max_samples=$(extract_config_value "$config" "max_samples")
    
    # Apply user overrides
    vis_interval=${VIS_INTERVAL:-$vis_interval}
    gpus=${OVERRIDE_GPUS:-$gpus}
    batch_size=${OVERRIDE_BATCH_SIZE:-$batch_size}
    
    # Resource allocation based on experiment type
    local mem_per_cpu="16G"
    local cpus_per_task="16"
    local tmp_space="800G"
    
    if [[ "$experiment_type" == "minimal" ]]; then
        mem_per_cpu="8G"
        cpus_per_task="8"
        tmp_space="200G"
    elif [[ "$experiment_type" == "debug" ]]; then
        mem_per_cpu="12G"
        cpus_per_task="12"
        tmp_space="400G"
    fi
    
    cat << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${account}
#SBATCH --time=${time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --tmp=${tmp_space}
#SBATCH --gpus=rtx_4090:${gpus}
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${output_dir}/logs/slurm-%j.out
#SBATCH --error=${output_dir}/logs/slurm-%j.err

set -euo pipefail

# =============================================================================
# JOB INITIALIZATION AND LOGGING
# =============================================================================

echo "=========================================================================="
echo "PANGAEA-bench SLURM Job v3.0"
echo "Job: \${SLURM_JOB_ID} | Node: \${SLURMD_NODENAME} | Started: \$(date)"
echo "Model: ${encoder} | Experiment: ${experiment_type} | Seed: ${seed}"
echo "GPUs: ${gpus} | Epochs: ${epochs} | LR: ${learning_rate} | Batch: ${batch_size}"
echo "Debug Mode: ${debug_mode} | Vis Interval: ${vis_interval}"
echo "=========================================================================="

# Module loading
log_info() { echo "[INFO \$(date '+%H:%M:%S')] \$1"; }
log_error() { echo "[ERROR \$(date '+%H:%M:%S')] \$1"; }
log_info "Loading required modules..."
module load eth_proxy stack/2024-06 openblas/0.3.24

# =============================================================================
# WORKSPACE SETUP WITH FULL ISOLATION
# =============================================================================

log_info "Setting up isolated workspace in TMPDIR..."

# Create workspace structure
mkdir -p "\$TMPDIR/agbd_data" "\$TMPDIR/agbd_splits" "\$TMPDIR/pangaea_workspace" "\$TMPDIR/logs"

# Sync data (optimized for different experiment types)
log_info "Syncing AGBD data to TMPDIR..."
if [[ "${debug_mode}" == "true" ]]; then
    # For debug/minimal: sync only essential files for speed
    log_info "Debug mode: syncing minimal dataset..."
    rsync --include '*v4_*-20.h5' --exclude '*' -av /cluster/work/igp_psr/gsialelli/Data/patches/ "\$TMPDIR/agbd_data/" --max-size=100M || true
else
    # For production: sync full dataset
    log_info "Production mode: syncing full dataset..."
    rsync --include '*v4_*-20.h5' --exclude '*' -av /cluster/work/igp_psr/gsialelli/Data/patches/ "\$TMPDIR/agbd_data/"
fi

# Sync split files and metadata
for split in train val test; do
    rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/\${split}_features_2019.csv "\$TMPDIR/agbd_data/" || log_error "Failed to sync \${split} CSV"
done
rsync -av /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl "\$TMPDIR/agbd_data/" || log_error "Failed to sync statistics"
rsync -av /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl "\$TMPDIR/agbd_splits/" || log_error "Failed to sync biomes splits"

# =============================================================================
# CODEBASE SYNCHRONIZATION (NEW IN v3.0)
# =============================================================================

log_info "Syncing PANGAEA-bench codebase to TMPDIR..."
PANGAEA_HOME="${pangaea_home}"
VENV_HOME="${venv_home}"
TMPDIR_PANGAEA="\$TMPDIR/pangaea_workspace"

# Sync entire codebase with exclusions for performance
rsync -av --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'wandb' --exclude 'test_agbd_logs' "\$PANGAEA_HOME/" "\$TMPDIR_PANGAEA/"

# Verify critical files
if [[ ! -f "\$TMPDIR_PANGAEA/pangaea/run.py" ]]; then
    log_error "Critical file missing: pangaea/run.py"
    exit 1
fi

if [[ ! -f "\$TMPDIR_PANGAEA/agbd_visualization.py" ]]; then
    log_error "Critical file missing: agbd_visualization.py"
    exit 1
fi

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP (ALWAYS COPY FOR COMPLETE ISOLATION)
# =============================================================================

log_info "Syncing virtual environment to TMPDIR for complete isolation..."
TMPDIR_VENV="\$TMPDIR/pangaea_venv"
rsync -av "\$VENV_HOME/" "\$TMPDIR_VENV/"

# Change to workspace and activate copied environment
cd "\$TMPDIR_PANGAEA"
source "\$TMPDIR_VENV/bin/activate"

# Install visualization dependencies if missing
python -c "import matplotlib" 2>/dev/null || pip install matplotlib seaborn

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { log_error "PyTorch not available"; exit 1; }
python -c "import sys; sys.path.append('.'); import agbd_visualization; print('AGBD visualization module loaded')" || { log_error "AGBD visualization not available"; exit 1; }

# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================

PERSISTENT_OUTPUT_DIR="${output_dir}/results/\${SLURM_JOB_ID}_${encoder}_seed${seed}_\$(date +%Y%m%d_%H%M%S)_${experiment_type}"
mkdir -p "\$PERSISTENT_OUTPUT_DIR"

# Save job metadata
echo "${encoder}" > "\$PERSISTENT_OUTPUT_DIR/encoder.txt"
echo "${seed}" > "\$PERSISTENT_OUTPUT_DIR/seed.txt"
echo "${epochs}" > "\$PERSISTENT_OUTPUT_DIR/epochs.txt"
echo "${learning_rate}" > "\$PERSISTENT_OUTPUT_DIR/learning_rate.txt"
echo "${experiment_type}" > "\$PERSISTENT_OUTPUT_DIR/experiment_type.txt"
echo "\$(date)" > "\$PERSISTENT_OUTPUT_DIR/start_time.txt"

# =============================================================================
# TRAINING COMMAND CONSTRUCTION
# =============================================================================

log_info "Constructing training command..."

# Base command with adaptive GPU configuration
if [[ ${gpus} -eq 1 ]]; then
    # Single GPU mode (for minimal/debug)
    TRAIN_CMD="python pangaea/run.py"
else
    # Multi-GPU mode (for production)
    TRAIN_CMD="torchrun --nnodes=1 --nproc_per_node=${gpus} --rdzv-backend=c10d --rdzv-endpoint=localhost:0 pangaea/run.py"
fi

# Core configuration
TRAIN_CMD="\$TRAIN_CMD \\
    --config-name=train_agbd \\
    dataset=agbd \\
    encoder=${encoder} \\
    decoder=reg_upernet \\
    criterion=mse \\
    task=regression \\
    optimizer=adamw \\
    lr_scheduler=step_agbd"

# Training parameters
TRAIN_CMD="\$TRAIN_CMD \\
    task.trainer.n_epochs=${epochs} \\
    task.trainer.ckpt_interval=1 \\
    task.trainer.eval_interval=1 \\
    seed=${seed} \\
    task.trainer.learning_rate=${learning_rate} \\
    task.trainer.batch_size=${batch_size}"

# Debug mode specific settings
if [[ "${debug_mode}" == "true" ]]; then
    TRAIN_CMD="\$TRAIN_CMD \\
        dataset.debug=true"
    
    # Minimal mode gets even more restrictive settings
    if [[ "${experiment_type}" == "minimal" ]]; then
        TRAIN_CMD="\$TRAIN_CMD \\
            dataset.max_samples=${max_samples} \\
            task.trainer.eval_every_n_steps=10"
    fi
else
    TRAIN_CMD="\$TRAIN_CMD \\
        dataset.debug=false"
fi

# Dataset and path configuration
TRAIN_CMD="\$TRAIN_CMD \\
    dataset.img_size=48 \\
    dataset.root_path=\$TMPDIR/agbd_data \\
    dataset.hdf5_dir=\$TMPDIR/agbd_data \\
    dataset.mapping_path=\$TMPDIR/agbd_splits \\
    dataset.norm_path=\$TMPDIR/agbd_data"

# Advanced features
TRAIN_CMD="\$TRAIN_CMD \\
    image_processing_strategy=padding \\
    use_padding_strategy=true \\
    central_pixel_scaling_enabled=true \\
    use_wandb=true \\
    task.evaluator.visualization_interval=${vis_interval}"

# Output configuration
TRAIN_CMD="\$TRAIN_CMD \\
    work_dir=\$PERSISTENT_OUTPUT_DIR \\
    hydra.run.dir=\$PERSISTENT_OUTPUT_DIR/hydra_outputs/\$(date +%Y-%m-%d)/\$(date +%H-%M-%S)"

# =============================================================================
# TRAINING EXECUTION WITH ERROR HANDLING
# =============================================================================

log_info "Starting training with command:"
echo "\$TRAIN_CMD" | tee "\$PERSISTENT_OUTPUT_DIR/command.txt"

# Start training with comprehensive error handling
START_TIME=\$(date +%s)
echo "running" > "\$PERSISTENT_OUTPUT_DIR/status.txt"

set +e  # Temporarily disable exit on error for proper cleanup
eval "\$TRAIN_CMD" > "\$PERSISTENT_OUTPUT_DIR/training.log" 2>&1
EXIT_CODE=\$?
set -e

END_TIME=\$(date +%s)
TRAINING_TIME=\$((END_TIME - START_TIME))

# Save execution metadata
echo "\$EXIT_CODE" > "\$PERSISTENT_OUTPUT_DIR/exit_code.txt"
echo "\$TRAINING_TIME" > "\$PERSISTENT_OUTPUT_DIR/training_time_seconds.txt"
echo "\$(date)" > "\$PERSISTENT_OUTPUT_DIR/end_time.txt"

# =============================================================================
# JOB COMPLETION AND CLEANUP
# =============================================================================

if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "success" > "\$PERSISTENT_OUTPUT_DIR/status.txt"
    log_info "Training completed successfully in \${TRAINING_TIME}s"
    
    # Copy important artifacts back
    if [[ -d "\$TMPDIR_PANGAEA/wandb" ]]; then
        rsync -av "\$TMPDIR_PANGAEA/wandb/" "\$PERSISTENT_OUTPUT_DIR/wandb_backup/" || true
    fi
    
else
    echo "failed" > "\$PERSISTENT_OUTPUT_DIR/status.txt"
    log_error "Training failed with exit code \$EXIT_CODE after \${TRAINING_TIME}s"
    
    # Copy debug information
    cp "\$TMPDIR_PANGAEA/pangaea/run.py" "\$PERSISTENT_OUTPUT_DIR/debug_run.py" || true
    cp "\$TMPDIR_PANGAEA/agbd_visualization.py" "\$PERSISTENT_OUTPUT_DIR/debug_agbd_visualization.py" || true
    env > "\$PERSISTENT_OUTPUT_DIR/debug_environment.txt" || true
fi

echo "=========================================================================="
echo "Job completed: \$(date)"
echo "Total time: \${TRAINING_TIME}s"
echo "Exit code: \$EXIT_CODE"
echo "Results: \$PERSISTENT_OUTPUT_DIR"
echo "=========================================================================="

exit \$EXIT_CODE
EOF
}

# ============================================================================
# JOB GENERATION WITH ENHANCED FEATURES
# ============================================================================

generate_single_job() {
    local encoder=$1
    local experiment_type=$2
    local seed=$3
    local account=$4
    local output_dir=$5
    local pangaea_home=$6
    local venv_home=$7
    local batch_size=$8
    
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local time_limit=$(extract_config_value "$config" "time")
    local epochs=$(extract_config_value "$config" "epochs")
    local learning_rate=$(extract_config_value "$config" "lr")
    
    local job_name="PANGAEA_v3_${encoder}_${experiment_type}_s${seed}"
    local script_file="${output_dir}/jobs/${job_name}.slurm"
    
    mkdir -p "$(dirname "$script_file")" "${output_dir}/logs"
    
    generate_slurm_template "$job_name" "$account" "$time_limit" "$output_dir" \
        "$pangaea_home" "$venv_home" "$encoder" "$epochs" "$seed" "$learning_rate" \
        "$experiment_type" "$batch_size" > "$script_file"
    
    echo "$script_file"
}

generate_batch_jobs() {
    local models_string=$1
    local experiment_type=$2
    local account=$3
    local output_dir=$4
    local pangaea_home=$5
    local venv_home=$6
    local batch_size=$7
    
    IFS=',' read -ra MODELS <<< "$models_string"
    local config=${EXPERIMENT_CONFIGS[$experiment_type]}
    local num_seeds=$(extract_config_value "$config" "seeds")
    
    local job_files=()
    local total_jobs=0
    
    log_info "Generating jobs for experiment type: $experiment_type"
    log_info "Configuration: $config"
    
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)  # Trim whitespace
        
        if ! validate_model "$model"; then
            log_warning "Skipping invalid model: $model"
            continue
        fi
        
        for ((seed=1; seed<=num_seeds; seed++)); do
            job_file=$(generate_single_job "$model" "$experiment_type" "$seed" \
                "$account" "$output_dir" "$pangaea_home" "$venv_home" "$batch_size")
            job_files+=("$job_file")
            total_jobs=$((total_jobs + 1))
            echo "Generated: $(basename "$job_file")"
        done
    done
    
    if [[ $total_jobs -eq 0 ]]; then
        log_error "No valid jobs generated"
        return 1
    fi
    
    log_success "Generated $total_jobs job files"
    
    # Create submission script
    local submit_script="${output_dir}/submit_all_jobs_v3.sh"
    cat > "$submit_script" << EOF
#!/bin/bash
# PANGAEA-bench v3.0 Batch Submission Script
# Generated: $(date)
# Experiment: $experiment_type
# Total jobs: $total_jobs

set -e

echo "Submitting $total_jobs PANGAEA-bench v3.0 jobs..."
echo "Experiment type: $experiment_type"
echo "Started: \$(date)"
echo ""

EOF
    
    for job_file in "${job_files[@]}"; do
        echo "echo \"Submitting: \$(basename $job_file)\"" >> "$submit_script"
        echo "sbatch $job_file" >> "$submit_script"
        echo "sleep 1  # Avoid overwhelming SLURM" >> "$submit_script"
    done
    
    cat >> "$submit_script" << EOF

echo ""
echo "All jobs submitted: \$(date)"
echo "Monitor with: squeue -u \$(whoami)"
echo "Results will be in: $output_dir/results/"
EOF
    
    chmod +x "$submit_script"
    
    # Output for main function
    echo "job_files=(${job_files[*]})"
    echo "submit_script=$submit_script"
    echo "total_jobs=$total_jobs"
}

# ============================================================================
# MODEL GROUPING FUNCTIONS
# ============================================================================

get_all_models() {
    echo "${AVAILABLE_MODELS[@]}" | tr ' ' ','
}

get_foundation_models() {
    echo "satmae_base,scalemae,prithvi,dofa,gfmswin,remoteclip,spectralgpt,satlasnet_mi,satlasnet_si,galileo"
}

get_ssl4eo_models() {
    echo "ssl4eo_mae_optical,ssl4eo_mae_sar,ssl4eo_data2vec,ssl4eo_dino,ssl4eo_moco"
}

get_croma_models() {
    echo "croma_joint,croma_optical,croma_sar"
}

get_baseline_models() {
    echo "resnet50_pretrained,resnet50_scratch,vit,vit_mi,vit_scratch,unet_encoder,unet_encoder_mi"
}

get_debug_models() {
    echo "satmae_base,dofa"  # Fast models for debugging
}

# ============================================================================
# STATUS AND MONITORING
# ============================================================================

show_job_status() {
    local output_dir=$1
    echo ""
    echo "PANGAEA-bench v3.0 Job Status"
    echo "=============================="
    echo ""
    
    # Current jobs
    echo "Active Jobs:"
    if command -v squeue >/dev/null 2>&1; then
        squeue -u $(whoami) --format="%.10i %.30j %.8T %.10M %.15P %.6D %R" 2>/dev/null || echo "  No jobs in queue"
    else
        echo "  SLURM not available (running locally?)"
    fi
    
    echo ""
    
    # Recent results with v3.0 format
    echo "Recent Results:"
    if [[ -d "$output_dir/results" ]]; then
        local count=0
        for result_dir in "$output_dir/results"/*; do
            if [[ -d "$result_dir" ]]; then
                local dirname=$(basename "$result_dir")
                local status="unknown"
                [[ -f "$result_dir/status.txt" ]] && status=$(cat "$result_dir/status.txt")
                
                echo "  $dirname [$status]"
                count=$((count + 1))
                
                # Show only recent 10 results
                if [[ $count -ge 10 ]]; then
                    echo "  ... (showing first 10 of many results)"
                    break
                fi
            fi
        done
        
        if [[ $count -eq 0 ]]; then
            echo "  No results found"
        fi
    else
        echo "  Results directory not found: $output_dir/results"
    fi
}

collect_results() {
    local output_dir=$1
    local summary_file="${output_dir}/results_summary_v3.csv"
    
    if [[ ! -d "$output_dir/results" ]]; then
        log_error "Results directory not found: $output_dir/results"
        return 1
    fi
    
    log_info "Collecting results from: $output_dir/results"
    
    # Enhanced CSV header for v3.0
    cat > "$summary_file" << EOF
job_id,encoder,seed,epochs,learning_rate,experiment_type,status,training_time_seconds,training_time_human,start_time,end_time,exit_code,result_directory,pangaea_version
EOF
    
    local total_count=0
    local n_success=0
    local n_failed=0
    local n_running=0
    
    for result_dir in "$output_dir/results"/*; do
        if [[ -d "$result_dir" ]]; then
            local dirname=$(basename "$result_dir")
            
            # Extract metadata with defaults
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
            local pangaea_version="v3.0"
            
            # Read metadata files
            [[ -f "$result_dir/encoder.txt" ]] && encoder=$(cat "$result_dir/encoder.txt")
            [[ -f "$result_dir/seed.txt" ]] && seed=$(cat "$result_dir/seed.txt")
            [[ -f "$result_dir/epochs.txt" ]] && epochs=$(cat "$result_dir/epochs.txt")
            [[ -f "$result_dir/learning_rate.txt" ]] && learning_rate=$(cat "$result_dir/learning_rate.txt")
            [[ -f "$result_dir/experiment_type.txt" ]] && experiment_type=$(cat "$result_dir/experiment_type.txt")
            [[ -f "$result_dir/status.txt" ]] && status=$(cat "$result_dir/status.txt")
            [[ -f "$result_dir/training_time_seconds.txt" ]] && training_time=$(cat "$result_dir/training_time_seconds.txt")
            [[ -f "$result_dir/start_time.txt" ]] && start_time=$(cat "$result_dir/start_time.txt")
            [[ -f "$result_dir/end_time.txt" ]] && end_time=$(cat "$result_dir/end_time.txt")
            [[ -f "$result_dir/exit_code.txt" ]] && exit_code=$(cat "$result_dir/exit_code.txt")
            
            # Convert training time to human readable
            if [[ -n "$training_time" ]] && [[ "$training_time" -gt 0 ]]; then
                training_time_human="$((training_time / 3600))h$((training_time % 3600 / 60))m$((training_time % 60))s"
            fi
            
            # Write to CSV
            echo "$job_id,$encoder,$seed,$epochs,$learning_rate,$experiment_type,$status,$training_time,$training_time_human,$start_time,$end_time,$exit_code,$result_dir,$pangaea_version" >> "$summary_file"
            
            total_count=$((total_count + 1))
            case "$status" in
                "success") n_success=$((n_success + 1)) ;;
                "failed") n_failed=$((n_failed + 1)) ;;
                "running") n_running=$((n_running + 1)) ;;
            esac
        fi
    done
    
    log_success "Results summary saved to: $summary_file"
    
    # Enhanced report for v3.0
    local report_file="${output_dir}/results_report_v3.txt"
    cat > "$report_file" << EOF
PANGAEA-bench v3.0 Results Report
=================================
Generated: $(date)

Summary:
--------
Total jobs: $total_count
Successful: $n_success
Failed:     $n_failed  
Running:    $n_running

Details:
--------
See $summary_file for complete information.

Top Performers:
EOF
    
    # Add top performers if we have successful results
    if [[ $n_success -gt 0 ]]; then
        echo "$(grep ",success," "$summary_file" | head -5)" >> "$report_file"
    else
        echo "No successful jobs yet." >> "$report_file"
    fi
    
    log_success "Results report saved to: $report_file"
    
    # Display summary
    echo ""
    echo "Results Summary:"
    echo "==============="
    echo "Total jobs: $total_count"
    echo "✅ Successful: $n_success"
    echo "❌ Failed: $n_failed"
    echo "🔄 Running: $n_running"
}

# ============================================================================
# HELP AND DOCUMENTATION
# ============================================================================

show_help() {
    cat << EOF

PANGAEA-bench SLURM Manager v3.0
================================

IMPROVEMENTS IN v3.0:
• Full codebase isolation via TMPDIR sync
• Always copy virtual environment for isolation
• Minimal debug mode for rapid development  
• Fine-grained parameter control
• Enhanced error handling and logging
• Adaptive resource allocation

USAGE:
  $0 [COMMAND] [OPTIONS]

COMMANDS:
  generate      Generate SLURM job files (without submission)
  submit        Generate and submit jobs to SLURM queue
  status        Show current job status and recent results
  collect       Collect and summarize all results
  list-models   List all available models
  help          Show this help message

MODEL SELECTION OPTIONS:
  -m, --models MODEL1,MODEL2,...    Specific models (comma-separated)
  -a, --all-models                  All available models
  -f, --foundation-models           Foundation models only
  -s, --ssl4eo-models               SSL4EO models only  
  -c, --croma-models                CROMA models only
  -b, --baseline-models             Baseline models only
  -d, --debug-models                Quick models for debugging

EXPERIMENT TYPES:
  -e, --experiment TYPE             Choose experiment configuration

  minimal:    1 epoch,  1 GPU,  50 samples,   vis every 20   (ultra-fast debug)
  debug:      1 epoch,  2 GPUs, 500 samples,  vis every 60   (quick validation)
  quick:      3 epochs, 4 GPUs, full dataset, vis every 500  (rapid testing)
  standard:   5 epochs, 8 GPUs, full dataset, vis every 1000 (default)
  thorough:   10 epochs, 8 GPUs, full dataset, vis every 2000 (comprehensive)
  production: 20 epochs, 8 GPUs, full dataset, vis every 5000 (publication)

FINE-GRAINED CONTROL:
  --vis-interval N                  Override visualization interval
  --batch-size N                    Override batch size (default: 32)
  --gpus N                          Override GPU count

PATHS AND CONFIGURATION:
  -o, --output-dir DIR              Output directory
  -p, --pangaea-home DIR            PANGAEA-bench source directory
  -v, --venv-home DIR               Virtual environment directory
  --account ACCOUNT                 SLURM account

EXAMPLES:

  # Minimal debug (fastest possible test)
  $0 submit -d -e minimal

  # Quick validation of specific models
  $0 submit -m satmae_base,dofa -e debug

  # Standard production run for foundation models  
  $0 submit -f -e standard

  # Custom configuration with fine-grained control
  $0 submit -m scalemae --vis-interval 100 --batch-size 64 -e thorough

  # Generate jobs without submitting
  $0 generate -f -e quick

MONITORING:

  # Check job status
  $0 status
  
  # Collect and analyze results
  $0 collect

NOTE: v3.0 syncs the entire codebase and virtual environment to TMPDIR 
for each job, ensuring complete isolation and preventing module import issues. 
Use 'minimal' or 'debug' experiment types for rapid development and testing.

EOF
}

list_models() {
    echo ""
    echo "Available Models in PANGAEA-bench:"
    echo "=================================="
    echo ""
    echo "Foundation Models ($(echo "$(get_foundation_models)" | tr ',' '\n' | wc -l)):"
    echo "  $(get_foundation_models)" | tr ',' '\n' | sed 's/^/  /'
    echo ""
    echo "SSL4EO Models ($(echo "$(get_ssl4eo_models)" | tr ',' '\n' | wc -l)):"
    echo "  $(get_ssl4eo_models)" | tr ',' '\n' | sed 's/^/  /'
    echo ""
    echo "CROMA Models ($(echo "$(get_croma_models)" | tr ',' '\n' | wc -l)):"
    echo "  $(get_croma_models)" | tr ',' '\n' | sed 's/^/  /'
    echo ""
    echo "Baseline Models ($(echo "$(get_baseline_models)" | tr ',' '\n' | wc -l)):"
    echo "  $(get_baseline_models)" | tr ',' '\n' | sed 's/^/  /'
    echo ""
    echo "Debug Models (fast for testing):"
    echo "  $(get_debug_models)" | tr ',' '\n' | sed 's/^/  /'
    echo ""
    echo "Total: ${#AVAILABLE_MODELS[@]} models"
}

# ============================================================================
# MAIN INTERFACE
# ============================================================================

main() {
    local command=""
    local models=""
    local experiment_type="$DEFAULT_EXPERIMENT"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local pangaea_home="$DEFAULT_PANGAEA_HOME"
    local venv_home="$DEFAULT_VENV_HOME"
    local account="$DEFAULT_ACCOUNT"
    local batch_size="$DEFAULT_BATCH_SIZE"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            generate|submit|status|collect|list-models|help)
                command="$1"
                shift
                ;;
            -m|--models)
                models="$2"
                shift 2
                ;;
            -a|--all-models)
                models="$(get_all_models)"
                shift
                ;;
            -f|--foundation-models)
                models="$(get_foundation_models)"
                shift
                ;;
            -s|--ssl4eo-models)
                models="$(get_ssl4eo_models)"
                shift
                ;;
            -c|--croma-models)
                models="$(get_croma_models)"
                shift
                ;;
            -b|--baseline-models)
                models="$(get_baseline_models)"
                shift
                ;;
            -d|--debug-models)
                models="$(get_debug_models)"
                shift
                ;;
            -e|--experiment)
                experiment_type="$2"
                shift 2
                ;;
            --vis-interval)
                export VIS_INTERVAL="$2"
                shift 2
                ;;
            --batch-size)
                batch_size="$2"
                shift 2
                ;;
            --gpus)
                export OVERRIDE_GPUS="$2"
                shift 2
                ;;
            -o|--output-dir)
                output_dir="$2"
                shift 2
                ;;
            -p|--pangaea-home)
                pangaea_home="$2"
                shift 2
                ;;
            -v|--venv-home)
                venv_home="$2"
                shift 2
                ;;
            --account)
                account="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Default to help if no command
    if [[ -z "$command" ]]; then
        command="help"
    fi
    
    # Validate experiment type for generation commands
    if [[ "$command" =~ ^(generate|submit)$ ]]; then
        if ! validate_experiment_type "$experiment_type"; then
            log_error "Invalid experiment type: $experiment_type"
            log_info "Available types: ${!EXPERIMENT_CONFIGS[@]}"
            exit 1
        fi
    fi
    
    # Execute command
    case $command in
        help)
            show_help
            ;;
        list-models)
            list_models
            ;;
        generate|submit)
            if [[ -z "$models" ]]; then
                log_error "No models specified. Use -m or model group options (-a, -f, -s, -c, -b, -d)"
                exit 1
            fi
            
            log_info "=========================================="
            log_info "PANGAEA-bench v3.0 Job Generator"
            log_info "=========================================="
            log_info "Command: $command"
            log_info "Models: $models"
            log_info "Experiment: $experiment_type (${EXPERIMENT_CONFIGS[$experiment_type]})"
            log_info "Output directory: $output_dir"
            log_info "PANGAEA home: $pangaea_home"
            log_info "Virtual env: $venv_home"
            log_info "Account: $account"
            log_info "Batch size: $batch_size"
            
            # Create output directory
            mkdir -p "$output_dir"
            
            # Generate jobs
            log_info "Generating batch jobs..."
            generate_batch_jobs "$models" "$experiment_type" "$account" "$output_dir" \
                "$pangaea_home" "$venv_home" "$batch_size" > /tmp/batch_output_v3.txt
            
            # Extract results
            submit_script=$(grep "submit_script=" /tmp/batch_output_v3.txt | cut -d'=' -f2)
            total_jobs=$(grep "total_jobs=" /tmp/batch_output_v3.txt | cut -d'=' -f2)
            
            if [[ "$command" == "submit" ]]; then
                log_info "Generated $total_jobs job files. Submitting to SLURM..."
                if bash "$submit_script"; then
                    log_success "Batch submission completed!"
                    echo ""
                    echo "Monitor progress with: $0 status"
                    echo "Collect results with: $0 collect"
                else
                    log_error "Batch submission failed"
                    exit 1
                fi
            else
                log_success "Generated $total_jobs job files."
                echo ""
                echo "To submit all jobs: bash $submit_script"
                echo "Or submit individually: cd $output_dir/jobs && sbatch *.slurm"
            fi
            ;;
        status)
            show_job_status "$output_dir"
            ;;
        collect)
            collect_results "$output_dir"
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main "$@"
