import pandas as pd
import numpy as np
import os
import shutil
from tqdm.auto import tqdm # Kept tqdm.auto for general use
import pickle
import network_utils # Assuming this is a custom utility, ensure it's available
import argparse
import networkx as nx # Moved import here as it's used in the function

# Global parameters (moved to top for clarity)
t_window = 210  # Training window in days
threshold = 100  # Performance degradation threshold (adjust as needed)
max_simulation_days = 104  # Maximum number of simulation days
topK = 20
num_nearest_neighbors = 5
num_latent_factors = 32
version = "full"
ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#ps = [0.5] # Uncomment to test with a single value

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Script to process networks.")
parser.add_argument("-rs", "--RecSys", type=str, required=True, help="Algorithm to use for the experiment.")
parser.add_argument("-s", "--Seed", type=int, required=True, help="Seed for the experiment.")
parser.add_argument("-c", "--City", type=str, required=True, help="City for the experiment.")
parser.add_argument("-k", "--k_days", type=int, required=True, help="retraining frequency")
parser.add_argument("-sliding", "--sliding_window", type=bool, required=False,
                    default=False, help="sliding window for retraining")

args = parser.parse_args()

algorithm = args.RecSys
seed = args.Seed
city = args.City
k_days = args.k_days  # Epoch length in days
sliding_window = args.sliding_window
print("sliding_window", sliding_window)
print(f"Running experiment with algorithm: {algorithm} and seed: {seed} and k_days: {k_days}")
print(f"sliding window: {sliding_window}")


# --- Function Definition ---
def create_colocation_networks(user_histories):
    """
    Creates co-location networks from user movement histories.

    Parameters:
    -----------
    user_histories : dict
        Dictionary where keys are user IDs and values are pandas DataFrames
        containing their movement histories

    Returns:
    --------
    dict
        Dictionary where keys are epochs and values are networkx Graph objects
    """
    # Initialize dictionary to store networks for each epoch
    epoch_networks = {}

    # First, create a dictionary to store venue visits per epoch
    epoch_venue_users = {}  # {epoch: {venue: set(users)}}

    # Process each user's history to populate epoch_venue_users
    for uid, history in user_histories.items():
        if 'epoch' not in history.columns:
            #print(f"User {uid} was never active")
            continue
        else:
            # Skip rows with NaN epochs
            history = history.dropna(subset=['epoch'])

            # Process each visit
            for _, row in history.iterrows():
                epoch = int(row['epoch'])
                venue = row['venueID']

                # Initialize epoch dict if needed
                if epoch not in epoch_venue_users:
                    epoch_venue_users[epoch] = {}

                # Initialize venue set if needed
                if venue not in epoch_venue_users[epoch]:
                    epoch_venue_users[epoch][venue] = set()

                # Add user to venue's visitor set
                epoch_venue_users[epoch][venue].add(uid)

    # --- CRITICAL FIX: Moved edge creation outside the user history loop ---
    # Now, create edges between co-located users for each epoch
    for epoch, venue_users in epoch_venue_users.items():
        if epoch not in epoch_networks: # Initialize graph for this epoch
            epoch_networks[epoch] = nx.Graph()

        for venue, users in venue_users.items():
            users = list(users)  # Convert set to list for indexing

            # Add edges between all pairs of users who visited this venue
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    # No need for self-loop check as range(i+1, ...) handles it
                    epoch_networks[epoch].add_edge(users[i], users[j])

    return epoch_networks


# --- Prepare fixed paths outside the loop for efficiency ---
# Updated path for figures (added seed level)
experiments_figures_path = f'../out/experiments/{version}/{seed}/figures/'
if not sliding_window:
    this_experiment_figures_path = experiments_figures_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}"
else:
    this_experiment_figures_path = experiments_figures_path + f"SLIDING_city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}"

# Ensure figures directory exists (created once)
if not os.path.exists(this_experiment_figures_path):
    os.makedirs(this_experiment_figures_path)
    print(f"Created directory: {this_experiment_figures_path}")


# Updated path for networks (added seed level)
experiments_networks_path = f'../out/experiments/{version}/{seed}/colocationnetworks/'
if not sliding_window:
    this_experiment_networks_path = experiments_networks_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}"
else:
    this_experiment_networks_path = experiments_networks_path + f"SLIDING_city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}"

# Ensure networks directory exists (created once)
if not os.path.exists(this_experiment_networks_path):
    os.makedirs(this_experiment_networks_path)
    print(f"Created directory: {this_experiment_networks_path}")

rec_prob2user_histories_7_days = {}
# --- Main Loop for Recommender Probabilities ---
for recommender_prob in tqdm(ps, desc="Processing Recommender Probabilities"):

    user_histories_7_days = {} # Re-initialize for each recommender_prob, as intended

    # Define path to user histories for the current recommender_prob
    experiments_output_path = f"../data/processed/experiments/{version}/{seed}/"
    # This path rightly includes recommender_prob in its structure
    if not sliding_window:
        user_histories_path = experiments_output_path + f"city_{city}__train_{t_window}__step_7__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{recommender_prob}/user_histories"
    else:
        user_histories_path = experiments_output_path + f"SLIDING_city_{city}__train_{t_window}__step_7__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{recommender_prob}/user_histories"

    # Check if the user histories path exists before trying to list files
    if not os.path.exists(user_histories_path):
        print(f"Warning: User histories path not found for recommender_prob {recommender_prob}: {user_histories_path}")
        continue # Skip this recommender_prob if data is missing

    files = os.listdir(user_histories_path)

    for file in files:
        # Read the file
        user_id = file.split("_")[0] # Assuming filename format is "USERID_..."
        file_path = os.path.join(user_histories_path, file) # Use os.path.join for robustness
        user_histories_7_days[user_id] = pd.read_csv(file_path, index_col=0)
    rec_prob2user_histories_7_days[recommender_prob] = user_histories_7_days

# --- Main Loop for Recommender Probabilities ---
for recommender_prob in tqdm(ps, desc="Processing Recommender Probabilities"):

    user_histories = {} # Re-initialize for each recommender_prob, as intended

    # Define path to user histories for the current recommender_prob
    experiments_output_path = f"../data/processed/experiments/{version}/{seed}/"
    # This path rightly includes recommender_prob in its structure
    if not sliding_window:
        user_histories_path = experiments_output_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{recommender_prob}/user_histories"
    else:
        user_histories_path = experiments_output_path + f"SLIDING_city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{recommender_prob}/user_histories"

    # Check if the user histories path exists before trying to list files
    if not os.path.exists(user_histories_path):
        print(f"Warning: User histories path not found for recommender_prob {recommender_prob}: {user_histories_path}")
        continue # Skip this recommender_prob if data is missing

    files = os.listdir(user_histories_path)

    for file in files:
        # Read the file
        user_id = file.split("_")[0] # Assuming filename format is "USERID_..."
        file_path = os.path.join(user_histories_path, file) # Use os.path.join for robustness
        temp_df = pd.read_csv(file_path, index_col=0)
        if 'epoch' in rec_prob2user_histories_7_days[recommender_prob][user_id]:
            # IMPORTANT STEP: NORMALIZE EPOCH COUNT WITH RESPECT TO THE BASELINE
            temp_df['epoch'] = rec_prob2user_histories_7_days[recommender_prob][user_id]['epoch']
        user_histories[user_id] = temp_df

    # Create networks for the current set of user histories
    networks = create_colocation_networks(user_histories)

    # Save the networks
    for epoch, network in networks.items():
        network_filename = f"rec_{recommender_prob}_colocation_{epoch}.gpickle"
        full_save_path = os.path.join(this_experiment_networks_path, network_filename)
        with open(full_save_path, "wb") as f:
            pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Script execution completed.")