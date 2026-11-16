#!/bin/bash

# ============================================================================
# Velocity Ramping Training Script for ExoRL
# ============================================================================
# This script trains the policy with progressive velocity ramping.
# - First, learn to stand stable for 2000 iterations (no walking)
# - Then, every 500 iterations increase target x velocity by 0.1 m/s
# - Continue until reaching 1.5 m/s x velocity
# - Sensor noise is always enabled throughout training
# ============================================================================

# Configuration Variables
# ============================================================================

# Isaac Lab path
ISAAC_LAB_PATH="/workspace/isaaclab"

# Training parameters
TASK="Template-Exorl-Direct-v0"
NUM_ENVS=10000

# Training stage iterations
ITERATIONS_STANDING=1000       # First stage: learn to stand (no walking)
ITERATIONS_PER_VELOCITY=250    # Iterations per velocity increment

# Velocity progression: start at 0.0, increment by 0.1 until reaching 1.5
# 0.0 (standing only)
# 0.1, 0.2, 0.3, ..., 1.5
# That's 16 stages total (standing + 15 velocity increments)

# Base directories
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
LOGS_DIR="${WORKSPACE_ROOT}/source/exoRL/logs/rsl_rl/exorl_ppo_experiment"
SCRIPTS_DIR="${WORKSPACE_ROOT}/scripts/rsl_rl"

# Log file configuration
LOG_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "${WORKSPACE_ROOT}/logs"
LOG_FILE="${WORKSPACE_ROOT}/logs/velocity_ramping_training_${LOG_TIMESTAMP}.log"

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

# Run a training stage with optional walking velocity
run_training_stage() {
    local stage_num=$1
    local walk_speed_x=$2
    local max_iterations=$3
    local prev_run_name=$4
    
    local stage_description="Standing Only"
    if awk -v v="$walk_speed_x" 'BEGIN { exit !(v > 0.0) }'; then
        stage_description="Walking at $walk_speed_x m/s"
    fi
    
    echo ""
    echo "=========================================="
    echo "Stage $stage_num: $stage_description"
    echo "=========================================="
    echo "Target x velocity: $walk_speed_x m/s"
    echo "Max Iterations: $max_iterations"
    echo "Previous Run Name: ${prev_run_name:-'None (Fresh start)'}"
    echo "Sensor Noise: ON"
    echo "=========================================="
    echo ""
    
    # Build the training command
    local cmd="${ISAAC_LAB_PATH}/isaaclab.sh -p ${SCRIPTS_DIR}/train.py"
    cmd="$cmd --task=${TASK}"
    cmd="$cmd --headless"
    cmd="$cmd --log_sub_rewards"
    cmd="$cmd --sensor_noise"
    cmd="$cmd --num_envs=${NUM_ENVS}"
    cmd="$cmd --max_iterations=${max_iterations}"
    
    # Add walking parameters if velocity > 0
    if awk -v v="$walk_speed_x" 'BEGIN { exit !(v > 0.0) }'; then
        cmd="$cmd --walk_speed_x ${walk_speed_x}"
        cmd="$cmd --walk_speed_y 0.0"
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
echo "║    ExoRL Velocity Ramping Training - Starting Pipeline     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Isaac Lab Path: ${ISAAC_LAB_PATH}"
echo "  Task: ${TASK}"
echo "  Num Envs: ${NUM_ENVS}"
echo "  Logs Directory: ${LOGS_DIR}"
echo "  Initial Standing Iterations: ${ITERATIONS_STANDING}"
echo "  Iterations per Velocity Increment: ${ITERATIONS_PER_VELOCITY}"
echo "  Target Velocity Range: 0.0 - 1.5 m/s"
echo ""

# Arrays to track run names
declare -a RUN_NAMES

# Stage 1: Standing only (no walking)
echo "Starting Stage 1 (Standing - learning stability)..."
run_training_stage 1 0.0 "${ITERATIONS_STANDING}" ""
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Stage 1 failed. Aborting pipeline."
    exit 1
fi

RUN_NAMES[0]=$(get_latest_run_name)
if [[ -z "${RUN_NAMES[0]}" ]]; then
    echo "[ERROR] Could not find run folder from Stage 1"
    exit 1
fi
echo "[INFO] Stage 1 run name: ${RUN_NAMES[0]}"

# Progressive velocity ramping: 0.1 to 1.5 m/s in 0.1 m/s increments
STAGE=2
VELOCITY=0.1
while awk -v v="$VELOCITY" 'BEGIN { exit !(v <= 1.5) }'; do
    echo ""
    echo "Starting Stage $STAGE (Velocity ramping)..."
    run_training_stage $STAGE $VELOCITY "${ITERATIONS_PER_VELOCITY}" "${RUN_NAMES[$((STAGE-2))]}"
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Stage $STAGE failed. Aborting pipeline."
        exit 1
    fi
    
    RUN_NAMES[$((STAGE-1))]=$(get_latest_run_name)
    if [[ -z "${RUN_NAMES[$((STAGE-1))]}" ]]; then
        echo "[ERROR] Could not find run folder from Stage $STAGE"
        exit 1
    fi
    echo "[INFO] Stage $STAGE run name: ${RUN_NAMES[$((STAGE-1))]}"
    
    # Increment velocity and stage
    VELOCITY=$(awk -v v="$VELOCITY" 'BEGIN { printf "%.1f", v + 0.1 }')
    STAGE=$((STAGE + 1))
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         All Training Stages Completed Successfully!        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
FINAL_CHECKPOINT=$(find_latest_checkpoint)
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Final run name: $(get_latest_run_name)"
echo "Total stages completed: $((STAGE - 1))"
echo ""
