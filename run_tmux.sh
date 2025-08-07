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
city="nyc"
k=1
algorithms=("MultiVAE")

# Loop through each seed provided as an argument
for algorithm in "${algorithms[@]}"; do
    for seed in "$@"; do
        for rb in "${rb_values[@]}"; do
            rb_safe=$(echo "$rb" | sed 's/\./p/g')
            session_name="${seed}__${rb_safe}__alg_${algorithm}"
            echo "Launching tmux session: $session_name"
            
            # Create a new detached tmux session and run the command inside
            tmux new-session -d -s "$session_name" "python main.py -c $city -s $seed -rb $rb -tw $train_days -sd $max_sim_days -rs $algorithm -k $k"
        done
    done
done

echo "All tmux sessions launched!"
