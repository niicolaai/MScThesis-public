#!/bin/bash

ENV_COUNTS=(1 10 100 1000 5000 10000 20000 30000 40000)
OUT_DIR="bench_results"
SAMPLE_INTERVAL=1  # seconds between samples

mkdir -p "$OUT_DIR"

for N in "${ENV_COUNTS[@]}"; do
    echo "=== Running num_envs=$N ==="

    LOG="$OUT_DIR/envs_${N}.csv"
    echo "timestamp,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb,cpu_percent,ram_used_mb,ram_total_mb" > "$LOG"

    # Start background monitor
    (
        while true; do
            TIMESTAMP=$(date +%s)
            GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits | head -1 | tr -d ' ')
            CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2+$4}')
            RAM=$(free -m | awk '/Mem:/{print $3","$2}')
            echo "$TIMESTAMP,$GPU,$CPU,$RAM" >> "$LOG"
            sleep "$SAMPLE_INTERVAL"
        done
    ) &
    MONITOR_PID=$!

    # Run training
    /workspace/isaaclab/isaaclab.sh -p /exoWE2/exoRL/scripts/rsl_rl/train.py \
        --task=Template-Exorl-Direct-v0 \
        --num_envs=$N \
        --max_iterations=20 \
        --log_sub_rewards \
        --headless

    # Stop monitor
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null

    echo "  -> Saved to $LOG"
    sleep 5  # cooldown between runs
done

echo "=== Done. Results in $OUT_DIR/ ==="