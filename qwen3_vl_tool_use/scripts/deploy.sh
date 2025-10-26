#!/bin/bash

MODEL=$1
NAME=$2
CUDA_DEVICES=${3:-"0,1,2,3"}  # Default to GPU 4,5,6,7 if not provided
PORT=${4:-10010}  # Default starting port is 10010 if not provided

# Store all background process PIDs
PIDS=()

# Signal handling function
cleanup() {
    echo "Stopping all servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill -TERM "$pid"
        fi
    done
    
    # Wait for processes to exit
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -KILL "$pid"
        fi
    done
    
    exit 0
}

# Set signal handlers
trap cleanup SIGINT SIGTERM

# Split CUDA_DEVICES by comma into array
IFS=',' read -ra DEVICE_ARRAY <<< "$CUDA_DEVICES"

# Iterate over device array and start server for each device
for ((i=0; i<${#DEVICE_ARRAY[@]}-1; i++)); do
    DEVICE=${DEVICE_ARRAY[$i]}
    CURRENT_PORT=$((PORT + DEVICE))
    
    echo "Starting server on GPU $DEVICE with port $CURRENT_PORT"
    CUDA_VISIBLE_DEVICES=$DEVICE python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $CURRENT_PORT \
        --model $MODEL \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.7 \
        --limit-mm-per-prompt.image 48 \
        --limit-mm-per-prompt.video 1 \
        --disable_mm_preprocessor_cache \
        --served-model-name $NAME &
    
    # Record process PID
    PIDS+=($!)
done

# The last device does not use &, but still needs to record PID
LAST_DEVICE=${DEVICE_ARRAY[-1]}
LAST_PORT=$((PORT + LAST_DEVICE))

echo "Starting server on GPU $LAST_DEVICE with port $LAST_PORT"
CUDA_VISIBLE_DEVICES=$LAST_DEVICE python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port $LAST_PORT \
    --model $MODEL \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.7 \
    --limit-mm-per-prompt.image 48 \
    --limit-mm-per-prompt.video 1 \
    --disable_mm_preprocessor_cache \
    --served-model-name $NAME &

# Record the last process PID
PIDS+=($!)

# Wait for all processes
wait


# bash scripts/deploy.sh /vepfs_c/gaolei/Qwen/Qwen3-VL-8B-Instruct REVPT_models 0,1,2,3
# python agent_eval.py --model-name REVPT_models --dataset vsibench --path /vepfs_c/gaolei/VSI-Bench/my_processed_data/train.parquet --port-pool 10010,10011,10012,10013