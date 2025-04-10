#!/bin/bash

# Define lists
#algorithms=("UserKNN" "LightGCN" "Neural_BPRMF" "MF" "CatRec" "CPop" "MultiVAE" "ItemKNN")
algorithms=("PGN")
seeds=(500 501 502 503 504)

# Loop through combinations
for seed in "${seeds[@]}"; do
  for algorithm in "${algorithms[@]}"; do
    echo "Running with seed=$seed and algorithm=$algorithm"
    python figures.py -rs "$algorithm" -s "$seed"
  done
done
