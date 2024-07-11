#!/bin/bash
SWEEP_ID="inease/Wandb-Showcase/tff0mff0"
NUM_AGENTS=3
PIDS=()

# Function to clean up all background processes
cleanup() {
    echo "Terminating all background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    echo "All background processes terminated."
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

for i in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent $SWEEP_ID > agent_$i.log 2>&1 &
    PIDS+=($!)  # Store the PID of the background process
    echo "Agent $i started with PID ${PIDS[-1]}"
done

# Wait for all background processes to complete
wait

# Remove the trap before exiting the script
trap - SIGINT
echo "Script completed."
