#!/bin/bash

# Get the latest 22 subdirectories and find the latest .pt file in each
MAIN_DIR="/home/nicolai/Documents/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment"

# Check if main directory exists
if [[ ! -d "$MAIN_DIR" ]]; then
    echo "Error: Directory $MAIN_DIR does not exist"
    exit 1
fi

# Get the latest 6 subdirectories (sorted by modification time, newest first)
mapfile -t DIRS < <(ls -dt "$MAIN_DIR"/*/ 2>/dev/null | head -6)

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "Error: No subdirectories found in $MAIN_DIR"
    exit 1
fi

# For each directory, find the latest .pt file and output its full path starting from /exoWE2
for dir in "${DIRS[@]}"; do
    # Find the latest .pt file in this directory
    latest_pt=$(find "$dir" -maxdepth 1 -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [[ -n "$latest_pt" ]]; then
        # Replace the prefix to start from /exoWE2
        echo "$latest_pt" | sed 's|.*/Documents/exoWE2|/exoWE2|'
    fi
done
