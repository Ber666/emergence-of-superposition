#!/bin/bash

# Function to generate a random port between 10000-65535
find_random_port() {
    # Generate random port between 10000 and 65535
    echo $((RANDOM % 55536 + 10000))
}

# Function to find idle GPUs
find_idle_gpus() {
    local num_gpus=$1
    local idle_gpus=()
    
    # Get GPU utilization using nvidia-smi
    local gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        echo "Error: nvidia-smi not available or no GPUs found" >&2
        exit 1
    fi
    
    # Parse GPU info and find idle GPUs
    while IFS=',' read -r gpu_id utilization memory; do
        # Remove spaces and convert to integers
        gpu_id=$(echo $gpu_id | tr -d ' ')
        utilization=$(echo $utilization | tr -d ' ')
        memory=$(echo $memory | tr -d ' ')
        
        # Consider GPU idle if utilization < 5% and memory < 1000MB
        if [ "$utilization" -lt 5 ] && [ "$memory" -lt 1000 ]; then
            idle_gpus+=($gpu_id)
            echo "Found idle GPU: $gpu_id (utilization: ${utilization}%, memory: ${memory}MB)" >&2
        fi
    done <<< "$gpu_info"
    
    # Check if we have enough idle GPUs
    if [ ${#idle_gpus[@]} -lt $num_gpus ]; then
        echo "Error: Only found ${#idle_gpus[@]} idle GPUs, but need $num_gpus" >&2
        echo "Available GPUs: ${idle_gpus[*]}" >&2
        exit 1
    fi
    
    # Return the first $num_gpus idle GPUs
    echo "${idle_gpus[@]:0:$num_gpus}"
}

# Check if argument file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <argument_file>"
    echo "Example: $0 args/sprosqa_coconut_graph_2l_8h_768d_fixed_bug_mix_no_answer_coconut_0.0_50ep.yaml"
    exit 1
fi

ARG_FILE=$1

# Check if argument file exists
if [ ! -f "$ARG_FILE" ]; then
    echo "Error: Argument file '$ARG_FILE' not found" >&2
    exit 1
fi

# Generate random port
PORT=$(find_random_port)
echo "Using random port: $PORT"

# Find 2 idle GPUs
echo "Searching for 2 idle GPUs..."
IDLE_GPUS=($(find_idle_gpus 2))

if [ $? -ne 0 ]; then
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES to the found idle GPUs
export CUDA_VISIBLE_DEVICES="${IDLE_GPUS[0]},${IDLE_GPUS[1]}"
echo "Using GPUs: ${IDLE_GPUS[0]}, ${IDLE_GPUS[1]}"

# Run the command with the random port and selected GPUs
torchrun --nnodes 1 --nproc_per_node 2 --master_port $PORT run.py "$ARG_FILE"