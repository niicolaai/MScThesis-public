#!/bin/bash

TARGET_DIR="/home/nicolai/Documents/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory does not exist: $TARGET_DIR"
    exit 1
fi

# Get the last 10 directories, sorted
echo "Last 10 folders with latest .pt files:"
echo "========================================"

find "$TARGET_DIR" -maxdepth 1 -type d -not -name '.' | sort | tail -10 | while read folder; do
    # Find the latest .pt file in this folder
    latest_pt=$(find "$folder" -maxdepth 1 -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$latest_pt" ]; then
        # Convert to path starting from /exoWE2
        relative_path=$(echo "$latest_pt" | sed 's|/home/nicolai/Documents/exoWE2|/exoWE2|')
        echo "$relative_path"
    fi
done
