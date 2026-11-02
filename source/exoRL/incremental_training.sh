#!/bin/bash

# ============================================================================
# Incremental Training Script for ExoRL
# ============================================================================
# This script trains the policy incrementally, progressively adding
# complexity through sensor noise, configuration randomization, and disturbance forces.
# ============================================================================

# Configuration Variables
# ============================================================================

# Isaac Lab path
ISAAC_LAB_PATH="/workspace/isaaclab"

# Training parameters
TASK="Template-Exorl-Direct-v0"
NUM_ENVS=10000

# Iterations for each training stage
ITERATIONS_STAGE_1=$((1*500))  # No augmentation
ITERATIONS_STAGE_2=$((1*250))   # Sensor noise
ITERATIONS_STAGE_3=$((1*1000))  # Sensor noise + Configuration randomization
# ITERATIONS_STAGE_4A=$((1*250))  # Light disturbances (0.5-2 N)
# ITERATIONS_STAGE_4B=$((1*250))  # Light-medium disturbances (1-5 N)
# ITERATIONS_STAGE_4C=$((1*250))  # Medium disturbances (2-10 N)
# ITERATIONS_STAGE_4D=$((1*250))  # Medium-heavy disturbances (5-15 N)
# ITERATIONS_STAGE_4E=$((1*250))  # Full disturbances (1-20 N)

# Base directories
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
LOGS_DIR="${WORKSPACE_ROOT}/source/exoRL/logs/rsl_rl/exorl_ppo_experiment"
SCRIPTS_DIR="${WORKSPACE_ROOT}/scripts/rsl_rl"

# Log file configuration
LOG_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${WORKSPACE_ROOT}/incremental_training_${LOG_TIMESTAMP}.log"

# ============================================================================
# Helper Functions
# ============================================================================

# Find the latest checkpoint and its run folder name
find_latest_checkpoint() {
    # Find the most recent folder (sorted by date)
    local latest_folder=$(ls -dt "${LOGS_DIR}"/*/ 2>/dev/null | head -1)
    
    if [[ -z "$latest_folder" ]]; then
        echo ""
        return 1
    fi
    
    # Check if exported/policy.pt exists
    local checkpoint_path="${latest_folder}exported/policy.pt"
    if [[ -f "$checkpoint_path" ]]; then
        echo "$checkpoint_path"
        return 0
    else
        echo ""
        return 1
    fi
}

# Get the latest run folder name (format: YYYY-MM-DD_HH-MM-SS)
get_latest_run_name() {
    local latest_folder=$(ls -dt "${LOGS_DIR}"/*/ 2>/dev/null | head -1)
    
    if [[ -z "$latest_folder" ]]; then
        echo ""
        return 1
    fi
    
    basename "$latest_folder" | sed 's:/$::'
}

# Run a training stage
run_training_stage() {
    local stage_num=$1
    local sensor_noise=$2
    local configuration_randomization=$3
    local disturbance_forces=$4
    local max_iterations=$5
    local prev_run_name=$6
    local disturbance_force_min=${7:-1.0}
    local disturbance_force_max=${8:-20.0}
    
    echo ""
    echo "=========================================="
    echo "Stage $stage_num Training"
    echo "=========================================="
    echo "Sensor Noise: $sensor_noise"
    echo "Configuration Randomization: $configuration_randomization"
    echo "Disturbance Forces: $disturbance_forces"
    echo "Max Iterations: $max_iterations"
    if [[ "$disturbance_forces" == "True" ]]; then
        echo "Disturbance Force Range: [$disturbance_force_min, $disturbance_force_max] N"
    fi
    echo "Previous Run Name: ${prev_run_name:-'None (Fresh start)'}"
    echo "=========================================="
    echo ""
    
    # Build the training command
    local cmd="${ISAAC_LAB_PATH}/isaaclab.sh -p ${SCRIPTS_DIR}/train.py"
    cmd="$cmd --task=${TASK}"
    cmd="$cmd --headless"
    cmd="$cmd --log_sub_rewards"
    cmd="$cmd --num_envs=${NUM_ENVS}"
    cmd="$cmd --max_iterations=${max_iterations}"
    
    # Add domain randomization flags only if they should be enabled
    if [[ "$sensor_noise" == "True" ]]; then
        cmd="$cmd --sensor_noise"
    fi
    if [[ "$configuration_randomization" == "True" ]]; then
        cmd="$cmd --configuration_randomization"
    fi
    if [[ "$disturbance_forces" == "True" ]]; then
        cmd="$cmd --disturbance_forces"
        cmd="$cmd --disturbance_force_min=${disturbance_force_min}"
        cmd="$cmd --disturbance_force_max=${disturbance_force_max}"
    fi
    
    # Add checkpoint loading if available
    if [[ -n "$prev_run_name" ]]; then
        cmd="$cmd --resume --load_run=${prev_run_name}"
        echo "[INFO] Will resume from run: $prev_run_name"
    fi
    
    echo "[INFO] Running command: $cmd"
    echo ""
    
    # Execute the training
    eval "$cmd"
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "[ERROR] Stage $stage_num training failed with exit code $exit_code"
        return 1
    fi
    
    echo ""
    echo "[INFO] Stage $stage_num training completed successfully"
    return 0
}

# ============================================================================
# Main Training Loop
# ============================================================================

# Redirect all output to both console and log file
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     ExoRL Incremental Training - Starting Pipeline         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Isaac Lab Path: ${ISAAC_LAB_PATH}"
echo "  Task: ${TASK}"
echo "  Num Envs: ${NUM_ENVS}"
echo "  Logs Directory: ${LOGS_DIR}"
echo ""

# Stage 1: No augmentation
echo "Starting Stage 1 (No augmentation)..."
run_training_stage 1 "False" "False" "False" "${ITERATIONS_STAGE_1}" ""
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Stage 1 failed. Aborting pipeline."
    exit 1
fi

# Find run name from Stage 1
RUN_NAME_1=$(get_latest_run_name)
if [[ -z "$RUN_NAME_1" ]]; then
    echo "[ERROR] Could not find run folder from Stage 1"
    exit 1
fi
echo "[INFO] Stage 1 run name: $RUN_NAME_1"

# Stage 2: Sensor noise only
echo ""
echo "Starting Stage 2 (Sensor noise only)..."
run_training_stage 2 "True" "False" "False" "${ITERATIONS_STAGE_2}" "$RUN_NAME_1"
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Stage 2 failed. Aborting pipeline."
    exit 1
fi

# Find run name from Stage 2
RUN_NAME_2=$(get_latest_run_name)
if [[ -z "$RUN_NAME_2" ]]; then
    echo "[ERROR] Could not find run folder from Stage 2"
    exit 1
fi
echo "[INFO] Stage 2 run name: $RUN_NAME_2"

# Stage 3: Sensor noise + Configuration randomization
echo ""
echo "Starting Stage 3 (Sensor noise + Configuration randomization)..."
run_training_stage 3 "True" "True" "False" "${ITERATIONS_STAGE_3}" "$RUN_NAME_2"
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Stage 3 failed. Aborting pipeline."
    exit 1
fi

# Find run name from Stage 3
RUN_NAME_3=$(get_latest_run_name)
if [[ -z "$RUN_NAME_3" ]]; then
    echo "[ERROR] Could not find run folder from Stage 3"
    exit 1
fi
echo "[INFO] Stage 3 run name: $RUN_NAME_3"

# # Stage 4: Sensor noise + Configuration randomization + Disturbance forces (Progressive, 5 sub-stages)
# echo ""
# echo "Starting Stage 4A (Light disturbances: 0.5-2 N)..."
# run_training_stage 4a "True" "True" "True" "${ITERATIONS_STAGE_4A}" "$RUN_NAME_3" 0.5 2.0
# if [[ $? -ne 0 ]]; then
#     echo "[ERROR] Stage 4A failed. Aborting pipeline."
#     exit 1
# fi

# RUN_NAME_4A=$(get_latest_run_name)
# if [[ -z "$RUN_NAME_4A" ]]; then
#     echo "[ERROR] Could not find run folder from Stage 4A"
#     exit 1
# fi
# echo "[INFO] Stage 4A run name: $RUN_NAME_4A"

# # Stage 4B: Light-medium disturbances
# echo ""
# echo "Starting Stage 4B (Light-medium disturbances: 1-5 N)..."
# run_training_stage 4b "True" "True" "True" "${ITERATIONS_STAGE_4B}" "$RUN_NAME_4A" 1.0 5.0
# if [[ $? -ne 0 ]]; then
#     echo "[ERROR] Stage 4B failed. Aborting pipeline."
#     exit 1
# fi

# RUN_NAME_4B=$(get_latest_run_name)
# if [[ -z "$RUN_NAME_4B" ]]; then
#     echo "[ERROR] Could not find run folder from Stage 4B"
#     exit 1
# fi
# echo "[INFO] Stage 4B run name: $RUN_NAME_4B"

# # Stage 4C: Medium disturbances
# echo ""
# echo "Starting Stage 4C (Medium disturbances: 2-10 N)..."
# run_training_stage 4c "True" "True" "True" "${ITERATIONS_STAGE_4C}" "$RUN_NAME_4B" 2.0 10.0
# if [[ $? -ne 0 ]]; then
#     echo "[ERROR] Stage 4C failed. Aborting pipeline."
#     exit 1
# fi

# RUN_NAME_4C=$(get_latest_run_name)
# if [[ -z "$RUN_NAME_4C" ]]; then
#     echo "[ERROR] Could not find run folder from Stage 4C"
#     exit 1
# fi
# echo "[INFO] Stage 4C run name: $RUN_NAME_4C"

# # Stage 4D: Medium-heavy disturbances
# echo ""
# echo "Starting Stage 4D (Medium-heavy disturbances: 5-15 N)..."
# run_training_stage 4d "True" "True" "True" "${ITERATIONS_STAGE_4D}" "$RUN_NAME_4C" 5.0 15.0
# if [[ $? -ne 0 ]]; then
#     echo "[ERROR] Stage 4D failed. Aborting pipeline."
#     exit 1
# fi

# RUN_NAME_4D=$(get_latest_run_name)
# if [[ -z "$RUN_NAME_4D" ]]; then
#     echo "[ERROR] Could not find run folder from Stage 4D"
#     exit 1
# fi
# echo "[INFO] Stage 4D run name: $RUN_NAME_4D"

# # Stage 4E: Full disturbances
# echo ""
# echo "Starting Stage 4E (Full disturbances: 1-20 N)..."
# run_training_stage 4e "True" "True" "True" "${ITERATIONS_STAGE_4E}" "$RUN_NAME_4D" 1.0 20.0
# if [[ $? -ne 0 ]]; then
#     echo "[ERROR] Stage 4E failed. Aborting pipeline."
#     exit 1
# fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         All Training Stages Completed Successfully!        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
FINAL_CHECKPOINT=$(find_latest_checkpoint)
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Final run name: $RUN_NAME_4E"
echo ""
echo "Training progression:"
echo "  Stage 1 (No augmentation): $RUN_NAME_1"
echo "  Stage 2 (Sensor noise): $RUN_NAME_2"
echo "  Stage 3 (Sensor noise + Config randomization): $RUN_NAME_3"
# echo "  Stage 4A (Light disturbances 0.5-2 N): $RUN_NAME_4A"
# echo "  Stage 4B (Light-medium disturbances 1-5 N): $RUN_NAME_4B"
# echo "  Stage 4C (Medium disturbances 2-10 N): $RUN_NAME_4C"
# echo "  Stage 4D (Medium-heavy disturbances 5-15 N): $RUN_NAME_4D"
# echo "  Stage 4E (Full disturbances 1-20 N): $RUN_NAME_4E"
