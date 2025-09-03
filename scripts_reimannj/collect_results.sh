#!/bin/bash


SEARCH_DIR="/scratch2/reimannj/pangaea-bench/test_agbd_logs"
OUTPUT_FILE="All_Results_pc96.csv"


echo "🔍 Starting result summarization (v11 - CSV output format)..."
echo "Input directory: ${SEARCH_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "--------------------------------------------------------"


echo "Encoder_Name,Job_ID,Val_RMSE,Test_RMSE,Duration,Last_Epoch,N_Epochs,Train_Patches,Finetune" > "${OUTPUT_FILE}"


for main_dir in "${SEARCH_DIR}"/*; do
    
    if [ -d "${main_dir}" ]; then
        
        
        full_dir_name=$(basename "${main_dir}")
        job_id=$(echo "${full_dir_name}" | grep -o '^[0-9]*')
        
        model_name=$(echo "${full_dir_name}" | sed 's/^[0-9]*_//' | sed 's/_full$//' | sed 's/_fast$//')

        
        for sub_dir in "${main_dir}"/*; do
            if [ -d "${sub_dir}" ]; then
                
                log_file=$(find "${sub_dir}" -name "train.log-*" -type f | sort -V | tail -n 1)

                
                sub_dir_name=$(basename "${sub_dir}")
                
                run_id="${model_name}_${sub_dir_name}"

                
                if [ -f "${log_file}" ]; then
                    echo "Processing -> ${run_id} (${job_id})"

                    
                    val_rmse=$(grep -A 4 "\[val\] ------- Centre Pixel MSE and RMSE --------" "${log_file}" | tail -n 5 | grep "RMSE" | tail -n 1 | awk '{print $2}' | xargs)

                    
                    test_rmse=$(grep -A 4 "\[test\] ------- Centre Pixel MSE and RMSE --------" "${log_file}" | tail -n 5 | grep "RMSE" | tail -n 1 | awk '{print $2}' | xargs)

                    
                    duration=""
                    
                    if [ -n "${test_rmse}" ] && [ "${test_rmse}" != "N/A" ]; then
                        duration=$(grep "\[test\] ------- Centre Pixel MSE and RMSE --------" "${log_file}" | tail -n 1 | awk -F' - ' '{print $3}' | awk '{print $1}' | xargs)
                    fi
                    
                    if [ -z "${duration}" ] || [ "${duration}" == "N/A" ]; then
                        duration=$(grep "\[val\] ------- Centre Pixel MSE and RMSE --------" "${log_file}" | tail -n 1 | awk -F' - ' '{print $3}' | awk '{print $1}' | xargs)
                    fi
                    
                    if [ -z "${duration}" ] || [ "${duration}" == "N/A" ]; then
                        duration=$(grep "Epoch \[" "${log_file}" | tail -n 1 | awk -F' - ' '{print $3}' | awk '{print $1}' | xargs)
                    fi

                    
                    n_epochs=$(grep "'n_epochs':" "${log_file}" | head -n 1 | awk -F': ' '{print $2}' | sed 's/,$//' | xargs)

                    
                    last_epoch=""
                    
                    last_started_epoch=$(grep "Starting epoch" "${log_file}" | tail -n 1 | sed -n 's/.*Starting epoch \([0-9]*\).*/\1/p')
                    
                    
                    if [ -n "${val_rmse}" ] && [ "${val_rmse}" != "N/A" ] && [ -n "${test_rmse}" ] && [ "${test_rmse}" != "N/A" ]; then
                        
                        if [ -n "${n_epochs}" ] && [ "${n_epochs}" != "N/A" ]; then
                            last_epoch=$((n_epochs - 1))  
                        fi
                    elif [ -n "${last_started_epoch}" ] && [ "${last_started_epoch}" != "" ]; then
                        
                        last_epoch="${last_started_epoch}"
                    else
                        
                        last_epoch=$(grep "Epoch \[" "${log_file}" | tail -n 1 | sed -n 's/.*Epoch \[\([0-9]*\).*/\1/p')
                    fi
                    
                    
                    train_patches=$(grep "Total number of train patches:" "${log_file}" | head -n 1 | awk '{print $NF}' | xargs)

                    
                    finetune_flag=$(grep "'finetune':" "${log_file}" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/,$//' | xargs)
                    
                    if [ "${finetune_flag}" = "True" ]; then
                        finetune_flag="Yes"
                    elif [ "${finetune_flag}" = "False" ]; then
                        finetune_flag="No"
                    else
                        finetune_flag="N/A"
                    fi

                    
                    epoch_progress=""
                    if [ -n "${last_epoch}" ] && [ -n "${n_epochs}" ]; then
                        epoch_progress="${last_epoch}/${n_epochs}"
                    elif [ -n "${last_epoch}" ]; then
                        epoch_progress="${last_epoch}/?"
                    elif [ -n "${n_epochs}" ]; then
                        epoch_progress="?/${n_epochs}"
                    fi

                    
                    if [ -z "${val_rmse}" ] && [ -z "${test_rmse}" ] && [ -z "${duration}" ] && [ -z "${train_patches}" ] && [ -z "${n_epochs}" ]; then
                        echo "Skipping -> ${run_id} (${job_id}) - Completely failed run (no data)"
                        continue
                    fi

                    
                    echo "\"${run_id}\",\"${job_id}\",\"${val_rmse:-N/A}\",\"${test_rmse:-N/A}\",\"${duration:-N/A}\",\"${epoch_progress:-N/A}\",\"${n_epochs:-N/A}\",\"${train_patches:-N/A}\",\"${finetune_flag:-N/A}\"" >> "${OUTPUT_FILE}"
                    
                else
                    echo "Processing -> ${run_id} (${job_id}) - No log file found"
                    
                    echo "\"${run_id}\",\"${job_id}\",\"N/A\",\"N/A\",\"N/A\",\"N/A\",\"N/A\",\"N/A\",\"N/A\"" >> "${OUTPUT_FILE}"
                fi
            fi
        done
    fi
done

cat "${OUTPUT_FILE}"