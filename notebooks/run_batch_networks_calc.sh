#!/bin/bash

#algorithms=("UserKNN" "LightGCN" "Neural_BPRMF" "MultiVAE" "ItemKNN" "MF")
algorithms=("MultiVAE")
seeds=(500 501 502 503 504)
k_days=(1)
city="nyc"

MAX_PARALLEL=32  # Set this to match your number of CPU cores

job_count=0
running_jobs=0
pids=()

total_jobs=$(( ${#algorithms[@]} * ${#seeds[@]} * ${#k_days[@]} ))
current_job=0

for algorithm in "${algorithms[@]}"; do
  for seed in "${seeds[@]}"; do
    for k_val in "${k_days[@]}"; do
      ((current_job++))
      echo "[$current_job / $total_jobs] Starting algorithm=$algorithm, seed=$seed, k=$k_val"

      python networks.py -rs "$algorithm" -s "$seed" -k "$k_val" -c "$city" &
      #python networks.py -rs "$algorithm" -s "$seed" -k "$k_val" -sliding "True" -c "$city" &

      pids+=($!)
      ((running_jobs++))

      if (( running_jobs >= MAX_PARALLEL )); then
        wait "${pids[@]}"
        pids=()
        running_jobs=0
      fi
    done
  done
done

# Wait for remaining jobs
wait "${pids[@]}"
echo "✅ All jobs completed."
