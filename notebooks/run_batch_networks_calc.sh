#!/bin/bash

# Define lists
#algorithms=("UserKNN" "LightGCN" "Neural_BPRMF"  "MultiVAE" "ItemKNN" "MF")
algorithms=("PGN")
seeds=(501 502)

# Loop through combinations
for seed in "${seeds[@]}"; do
  for algorithm in "${algorithms[@]}"; do
    echo "Running with seed=$seed and algorithm=$algorithm"
    python networks.py -rs "$algorithm" -s "$seed"
  done
done
