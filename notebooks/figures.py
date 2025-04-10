
import argparse
import pandas as pd
import pathlib
import numpy as np
import os
from tqdm.notebook import tqdm
import pickle
import networkx as nx
import collections


### SET HYPERPARAMETERS ###
city = "nyc"
t_window = 210  # Training window in days
k_days = 7  # Epoch length in days
threshold = 100  # Performance degradation threshold (adjust as needed)
max_simulation_days = 104  # Maximum number of simulation days
topK = 20
version = "full"
### ------------------- ###
parser = argparse.ArgumentParser(description="Script to process networks.")
parser.add_argument("-rs", "--RecSys", type=str, required=True, help="Algorithm to use for the experiment.")
parser.add_argument("-s", "--Seed", type=int, required=True, help="Seed for the experiment.")
args = parser.parse_args()

algorithm = args.RecSys
seed = args.Seed



basedir = pathlib.Path().cwd().parent

# Define base directory up to the version level
version_dir = basedir / 'out' / 'experiments' / version

simulated_temporal_colocation_net = {}

# Iterate over seed directories (0,1,...) inside the version directory
for seed_dir in version_dir.iterdir():
    #ensure it's a directory and only a number
    if not seed_dir.is_dir() or not seed_dir.name.isdigit():
        continue
    

    # Extract seed from folder name
    s = int(seed_dir.name)
    if s != seed:
        continue

    print(seed)
    # Define the colocation network simulation path inside each seed folder
    colocation_network_simulated_dir = (
        seed_dir / 'colocationnetworks' /
        f'city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}'
    )

    if not colocation_network_simulated_dir.exists():
        continue  # Skip if the directory doesn't exist

    for file_path in tqdm(colocation_network_simulated_dir.iterdir()):
        if file_path.name.endswith(".gpickle"):
            rec_prob = float(file_path.name.split("_")[1])
            epoch = int(file_path.name.split("_")[3].split(".")[0])

            # Initialize dictionary structure
            if seed not in simulated_temporal_colocation_net:
                simulated_temporal_colocation_net[seed] = {}
            if rec_prob not in simulated_temporal_colocation_net[seed]:
                simulated_temporal_colocation_net[seed][rec_prob] = {}

            # Load the graph
            with open(file_path, "rb") as f:
                simulated_temporal_colocation_net[seed][rec_prob][epoch] = pickle.load(f)




# Define probability values in a list for easy reference
probabilities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Create a dictionary to store all networks
colocation_nets = {p: simulated_temporal_colocation_net[seed][p] for p in probabilities}




def collapse_temporal_networks(networks):
    networks = list(networks.values())
    G = networks[0].copy()
    for i in range(1, len(networks)):
        G = nx.compose(G, networks[i])
    return G

agg_networks_dict = {
    0.0: collapse_temporal_networks(colocation_nets[0.0]),
    0.2: collapse_temporal_networks(colocation_nets[0.2]),
    0.4: collapse_temporal_networks(colocation_nets[0.4]),
    0.6: collapse_temporal_networks(colocation_nets[0.6]),
    0.8: collapse_temporal_networks(colocation_nets[0.8]),
    1.0: collapse_temporal_networks(colocation_nets[1.0])

}


with open(f"properties_{city}/nodes_coloc.csv", "a+") as f:
    f.seek(0)
    file_contents = f.read()
    if f"{seed}, {algorithm}" not in file_contents:
        for p, G in agg_networks_dict.items():
            f.write(f"{seed}, {algorithm}, {p} , {len(G.nodes)}\n")
        





def compute_degree_distribution(G):
    # Get degrees
    degrees = [d[1] for d in G.degree()]
    # Count frequency of each degree
    degree_count = collections.Counter(degrees)
    
    # Sort by degree
    x = sorted(degree_count.keys())
    y = [degree_count[k] / float(len(degrees)) for k in x]  # Normalized frequency
    
    return x, y

# Store all distribution data to find global min/max for both axes
all_distributions = []

# Debug information
for i, AggG in agg_networks_dict.items():
    x, y = compute_degree_distribution(AggG)
    print(len(x), len(y))
    all_distributions.append((x, y))



networks = list(agg_networks_dict.values())
ps = list(agg_networks_dict.keys())
for i, ((x, y), AggG) in enumerate(zip(all_distributions, networks)):
   
    # Optional: Add power-law fit if applicable
    # Log-transform for linear fit
    log_x = np.log10(np.array(x) + 1)  # +1 to handle k=0
    log_y = np.log10(np.array(y) + 1e-10)  # small constant to handle zeros
    
    # Simple linear regression for power-law exponent
    mask = np.isfinite(log_x) & np.isfinite(log_y)
    if np.sum(mask) >= 3:  # Need at least 3 points for meaningful fit
        z = np.polyfit(log_x[mask], log_y[mask], 1)
        p = np.poly1d(z)
        
        # Add fitted line
        x_fit = np.logspace(np.log10(min(x)), np.log10(max(x) + 1), 100)
        y_fit = 10**p(np.log10(x_fit))    
        
        # Add gamma value (power-law exponent)
        gamma = -z[0]  # Negative because P(k) ~ k^(-gamma)
        #write algorithm, p and gamma on the comparison.txt file

        # Write result to file
        with open(f"properties_{city}/gamma.csv", "a+") as f:
            f.seek(0)  # Move to beginning of file
            file_content = f.read()
            if f"{seed}, {algorithm}, {ps[i]}" not in file_content:
                 f.write(f"{seed}, {algorithm}, {ps[i]}, {gamma}\n")


def save_rich_club_density(networks_dict, rich_count=15, city="NYC", algorithm=None, output_file=None):


    # Make sure file directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process each network
    for key, network in networks_dict.items():
        # Get nodes and their degrees
        node_degrees = {node: network.degree(node) for node in network.nodes()}
        
        # Sort nodes by degree (highest first)
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Take top rich_count nodes (or all if fewer than rich_count)
        rich_nodes = [node for node, _ in sorted_nodes[:min(rich_count, len(sorted_nodes))]]
        
        # Calculate rich club density
        rich_club_density = 0
        if len(rich_nodes) > 1:
            # Count edges within rich club
            rich_edges = 0
            for i, u in enumerate(rich_nodes):
                for v in rich_nodes[i+1:]:
                    if network.has_edge(u, v):
                        rich_edges += 1
            
            # Calculate density
            potential_rich_edges = len(rich_nodes) * (len(rich_nodes) - 1) / 2
            rich_club_density = rich_edges / potential_rich_edges if potential_rich_edges > 0 else 0
        
        print(f"Rich club density for {seed} {algorithm} with p={key}: {rich_club_density:.3f}")
        # Write result to file
        with open(output_file, "a+") as f:
            f.seek(0)  # Move to beginning of file
            file_content = f.read()
            if f"{seed}, {algorithm}, {key}" not in file_content:
                f.write(f"{seed}, {algorithm}, {key}, {rich_club_density:.3f}\n")
            
    print(f"Rich club density values saved to {output_file}")

save_rich_club_density(agg_networks_dict, rich_count=15, city=city, algorithm=algorithm, output_file=f"properties_{city}/rich_club_density.csv")




from collections import defaultdict
import mobility_metrics_util

rog_by_p = {}
random_entropy_by_p = {}
unc_entropy_by_p = {}
real_entropy_by_p = {}
seq_based_entropy_by_p = {}
number_of_locations_by_p = {}
number_of_visits_by_p = {}
location_diversity_by_p = {}

collective_entropy_by_p = {}

lat_lon_by_p = defaultdict(set)

ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
new_columns = ['uid', 'lat', 'lon', 'venueID', 'venue_descr', 'time', 'city_name', 'First_Category', 'Second_Category', 'step', 'epoch', 'city', 'simulation_state']
for p in tqdm(ps):
    
    experiment_dir = f"../data/processed/experiments/{version}/{seed}/city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{p}"

    merged_user_histories = pd.DataFrame()
    
    for user_fname in pathlib.Path(experiment_dir).glob('user_histories/*.csv'):
        user_id = user_fname.stem.split('_')[0]
        user_history = pd.read_csv(user_fname, index_col=0)
        if 'simulation_state' not in user_history.columns:
            #print('skipping userid', user_id)
            continue
        user_history.columns = new_columns
        #keep only the rows that have simulation_state not Nan
        user_history = user_history[user_history['simulation_state'].notnull()]
        merged_user_histories = pd.concat([merged_user_histories, user_history], ignore_index=True)
        
    merged_user_histories.reset_index(drop=True, inplace=True)
    rog_by_p[p] = (mobility_metrics_util.radius_of_gyration(merged_user_histories).radius_of_gyration.values)
    unc_entropy_by_p[p] = mobility_metrics_util.uncorrelated_entropy(merged_user_histories).uncorrelated_entropy.values



# Write result to file
with open(f"properties_{city}/rog.csv", "a+")  as f:
    f.seek(0)  # Move to beginning of file
    file_content = f.read()
    if f"{seed}, {algorithm}" not in file_content:
        for p in ps:
            f.write(f"{seed}, {algorithm}, {p}, {np.mean(rog_by_p[p])/1000}\n")

# Write result to file
with open(f"properties_{city}/unc_entropy.csv", "a+")  as f:
    f.seek(0)  # Move to beginning of file
    file_content = f.read()
    if f"{seed}, {algorithm}" not in file_content:
        for p in ps:
            f.write(f"{seed}, {algorithm}, {p}, {np.mean(unc_entropy_by_p[p])}\n")



#global gini

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import contextily as ctx
import random

#set a seed for reproducibility
random.seed(42)
new_columns = ['uid', 'lat', 'lon', 'venueID', 'venue_descr', 'time', 'city_name', 'First_Category', 'Second_Category', 'step', 'epoch', 'city', 'simulation_state']

user_ids_per_p = []

for p in ps:
    experiment_dir = f"../data/processed/experiments/{version}/{seed}/city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{p}"
    user_files = list(Path(experiment_dir).glob('user_histories/*.csv'))
    user_ids = {fname.stem.split('_')[0] for fname in user_files}
    user_ids_per_p.append(user_ids)


# === STEP 2: Carica user histories e raccogli visite per ogni p ===
all_location_counts = {}

for p in tqdm(ps):
    experiment_dir = f"../data/processed/experiments/{version}/{seed}/city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{p}"
    merged_user_histories = pd.DataFrame()

    for user_fname in Path(experiment_dir).glob('user_histories/*.csv'):
        user_id = user_fname.stem.split('_')[0]
        user_history = pd.read_csv(user_fname, index_col=0)
        if 'simulation_state' not in user_history.columns:
            continue

        user_history.columns = new_columns
        user_history = user_history[user_history['simulation_state'].notnull()]
        merged_user_histories = pd.concat([merged_user_histories, user_history], ignore_index=True)

    location_counts = merged_user_histories.groupby(['lat', 'lon']).size().reset_index(name='visits')
    all_location_counts[p] = location_counts


counts_0 = all_location_counts[0.0]["visits"].values
counts_2 = all_location_counts[0.2]["visits"].values
counts_4 = all_location_counts[0.4]["visits"].values
counts_6 = all_location_counts[0.6]["visits"].values
counts_8 = all_location_counts[0.8]["visits"].values
counts_10 = all_location_counts[1.0]["visits"].values

def gini_index(values):
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    if n == 0 or np.sum(sorted_values) == 0:
        return 0
    
    indices = np.arange(1, n + 1)
    gini = (2 * np.sum(indices * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    return gini

gini_indices = {
    0.0: gini_index(counts_0),
    0.2: gini_index(counts_2),
    0.4: gini_index(counts_4),
    0.6: gini_index(counts_6),
    0.8: gini_index(counts_8),
    1.0: gini_index(counts_10)
}


with open(f"properties_{city}/gini_visits.csv", "a+") as f:
    f.seek(0)  # Move to beginning of file
    file_content = f.read()
    if f"{seed}, {algorithm}" not in file_content:
        for p, gini in gini_indices.items():
            f.write(f"{seed}, {algorithm}, {p}, {gini}\n")



#per each prob and rec, per each user calculate the gini of the number of visits and then average it
from collections import defaultdict

ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
new_columns = ['uid', 'lat', 'lon', 'venueID', 'venue_descr', 'time', 'city_name', 'First_Category', 'Second_Category', 'step', 'epoch', 'city', 'simulation_state']
#trajectories_dict is a dictionary in which the first level key is the probability p and the second level key is the algorithm. Then it stores the trajectories dataframe
trajectories_dict = defaultdict(lambda: defaultdict())

for p in tqdm(ps):
        
    experiment_dir = f"../data/processed/experiments/{version}/{seed}/city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{p}"
    merged_user_histories = pd.DataFrame()
    
    for user_fname in Path(experiment_dir).glob('user_histories/*.csv'):
        user_id = user_fname.stem.split('_')[0]
        user_history = pd.read_csv(user_fname, index_col=0)
        if 'simulation_state' not in user_history.columns:
            #print('skipping userid', user_id)
            continue
        user_history.columns = new_columns
        #keep only the rows that have simulation_state not Nan
        user_history = user_history[user_history['simulation_state'].notnull()]
        merged_user_histories = pd.concat([merged_user_histories, user_history], ignore_index=True)
        trajectories_dict[p][algorithm] = merged_user_histories

gini_dict = defaultdict(lambda: defaultdict(list))

for p in ps:
    # Get the trajectories for the current probability and algorithm
    trajectories = trajectories_dict[p][algorithm]
    for user in trajectories['uid'].unique():
        user_trajectory = trajectories[trajectories['uid'] == user]
        if(len(user_trajectory) == 0):
            continue
        # if p == 0.2 and user ==7:
        #     print(len((user_trajectory.groupby('venueID').size())), algorithm)
        # Calculate the Gini coefficient for the number of visits
        gini_coeff = gini_index(user_trajectory.groupby('venueID').size())
        gini_dict[p][algorithm].append(gini_coeff)
        


#clculate the mean gini for each algorithm and probability
gini_mean_dict = defaultdict(lambda: defaultdict(float))
for p in ps:
    gini_mean_dict[p][algorithm] = np.mean(gini_dict[p][algorithm])

# Write the results to a CSV file
with open(f"properties_{city}/gini_trajectories.csv", "a+") as f:
    f.seek(0)  # Move to beginning of file
    file_content = f.read()
    if f"{seed}, {algorithm}" not in file_content:
        for p in gini_mean_dict:
            # Get the Gini value for the current algorithm
            gini = gini_mean_dict[p][algorithm]
            # Write the data
            f.write(f"{seed}, {algorithm}, {p}, {gini}\n")