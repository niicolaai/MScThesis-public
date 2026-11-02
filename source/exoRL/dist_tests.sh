#!/bin/bash

# LOG_FILE="/exoWE2/exoRL/source/exoRL/disturbance_tests.log"
# exec 1> >(tee -a "$LOG_FILE")
# exec 2>&1

# Run disturbance tests with directions from 0 to 350 degrees in increments of 10

echo "Starting disturbance tests sweep from 0 to 350 degrees..."
echo ""

for direction in $(seq 0 10 350); do
    echo "Running disturbance tests with direction: ${direction}°"
    set -x
    /workspace/isaaclab/isaaclab.sh -p ../../scripts/rsl_rl/play.py \
        --task=Template-Exorl-Direct-v0 \
        --num_envs=10000 \
        --checkpoint=/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment/2026-04-25_14-37-04/model_1747.pt \
        --disturbance_tests \
        --direction $direction \
        --headless

    set +x
    
    if [ $? -ne 0 ]; then
        echo "Error running test with direction ${direction}°. Exiting."
        exit 1
    fi
    
    echo "Completed disturbance tests with direction: ${direction}°"
    echo ""
done

echo "All disturbance tests completed successfully!"


# --checkpoint=/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment/2026-04-25_14-37-04/model_1747.pt \
## --checkpoint=/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment/2026-04-26_02-21-34/model_1747.pt \
