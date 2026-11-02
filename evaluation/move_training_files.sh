#!/bin/bash

# ============================================================================
# STEP 1: CHECK FOR EXISTING NPZ FILES IN CURRENT DIRECTORY
# ============================================================================
CURRENT_NPZ_FILES=(*.npz)

if [ -e "${CURRENT_NPZ_FILES[0]}" ]; then
    echo "Found existing .npz files in current directory:"
    ls -lh *.npz
    echo ""
    read -p "Delete these files before copying new ones? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm *.npz
        echo "✓ Deleted existing .npz files"
    else
        echo "Cancelled"
        exit 0
    fi
fi

# ============================================================================
# STEP 2: COPY NEW NPZ FILES FROM LATEST TRAINING RUN
# ============================================================================
# Define the source path
SOURCE_PATH="/home/nicolai/Documents/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment"

# Check if source directory exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source directory not found: $SOURCE_PATH"
    exit 1
fi

# Find the latest directory (most recently modified)
LATEST_DIR=$(ls -dt "$SOURCE_PATH"/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No directories found in $SOURCE_PATH"
    exit 1
fi

echo "Found latest directory: $LATEST_DIR"

# Find all .npz files recursively and move them to current directory
NPZ_COUNT=$(find "$LATEST_DIR" -type f -name "*.npz" | wc -l)

if [ "$NPZ_COUNT" -eq 0 ]; then
    echo "No .npz files found in $LATEST_DIR"
    exit 0
fi

echo "Found $NPZ_COUNT .npz file(s). Copying to $(pwd)..."
find "$LATEST_DIR" -type f -name "*.npz" -exec cp {} . \;

echo "✓ Successfully copied all .npz files to $(pwd)"

# ============================================================================
# STEP 3: RUN EVALUATION SCRIPTS
# ============================================================================
echo ""
echo "Running evaluation scripts..."
echo ""

python3 eval_sub_rewards.py
python3 eval_play_data.py

echo ""
echo "✓ Evaluation complete"