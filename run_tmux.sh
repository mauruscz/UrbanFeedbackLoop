#!/bin/bash

# Check if at least one seed is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 seed1 [seed2 ...]"
    exit 1
fi

# Define the list of rb values
rb_values=(0.0 0.2 0.4 0.6 0.8 1.0)
train_days=210
max_sim_days=104
algorithm="PGN"

# Loop through each seed provided as an argument
for seed in "$@"; do
    for rb in "${rb_values[@]}"; do
        session_name="${seed}__${rb}__alg_${algorithm}"  # Create a unique session name
        echo "Launching tmux session: $session_name"
        
        # Create a new detached tmux session and run the command inside
        tmux new-session -d -s "$session_name" "python main.py -s $seed -rb $rb -tw $train_days -sd $max_sim_days -rs $algorithm"
    done
done

echo "All tmux sessions launched!"
