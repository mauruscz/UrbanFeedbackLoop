import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
import pickle
import network_utils
import argparse



user_histories = {}

city = "nyc"
t_window = 210  # Training window in days
k_days = 7  # Epoch length in days
threshold = 100  # Performance degradation threshold (adjust as needed)
max_simulation_days = 104  # Maximum number of simulation days
topK = 20
num_nearest_neighbors = 5
num_latent_factors = 32
version = "full"

parser = argparse.ArgumentParser(description="Script to process networks.")
parser.add_argument("-rs", "--RecSys", type=str, required=True, help="Algorithm to use for the experiment.")
parser.add_argument("-s", "--Seed", type=int, required=True, help="Seed for the experiment.")
args = parser.parse_args()

algorithm = args.RecSys
seed = args.Seed
print(f"Running experiment with algorithm: {algorithm} and seed: {seed}")

ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#ps = [0.5]



import networkx as nx
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
    
    # Process each user's history
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
                    epoch_networks[epoch] = nx.Graph()
                
                # Initialize venue set if needed
                if venue not in epoch_venue_users[epoch]:
                    epoch_venue_users[epoch][venue] = set()
                
                # Add user to venue's visitor set
                epoch_venue_users[epoch][venue].add(uid)
        
        # Create edges between co-located users
        for epoch, venue_users in epoch_venue_users.items():
            for venue, users in venue_users.items():
                users = list(users)  # Convert set to list for indexing
                
                # Add edges between all pairs of users who visited this venue
                for i in range(len(users)):
                    for j in range(i + 1, len(users)):
                        #avoid self loops
                        if users[i] == users[j]:
                            continue
                        epoch_networks[epoch].add_edge(users[i], users[j])
        
    return epoch_networks

colocation_networks = create_colocation_networks(user_histories)



import os
import pickle
import pandas as pd
from tqdm.notebook import tqdm

for recommender_prob in tqdm(ps):

    user_histories = {}


    # Updated experiment path to include seed level
    experiments_output_path = f"../data/processed/experiments/{version}/{seed}/"
    this_experiment_path = experiments_output_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{recommender_prob}"
    user_histories_path = this_experiment_path + "/user_histories"

    files = os.listdir(user_histories_path)

    for file in files:
        # Read the file
        user_id = file.split("_")[0]
        user_histories[user_id] = pd.read_csv(user_histories_path + '/' + file, index_col=0)

    networks = create_colocation_networks(user_histories)


    # Updated path for figures (added seed level)
    experiments_figures_path = f'../out/experiments/{version}/{seed}/figures/'
    this_experiment_figures_path = experiments_figures_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}"

    # Ensure directory exists
    if not os.path.exists(this_experiment_figures_path):
        os.makedirs(this_experiment_figures_path)


    # Updated path for networks (added seed level)
    experiments_networks_path = f'../out/experiments/{version}/{seed}/colocationnetworks/'
    this_experiment_networks_path = experiments_networks_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}"

    if not os.path.exists(this_experiment_networks_path):
        os.makedirs(this_experiment_networks_path)

    # Save the networks
    for epoch, network in networks.items():
        with open(
            this_experiment_networks_path + f"/rec_{recommender_prob}_colocation_{epoch}.gpickle",
            "wb"
        ) as f:
            pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
