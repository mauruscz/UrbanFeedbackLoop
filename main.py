import argparse
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model.simulation import Simulation
from model.recommender import UserKNN, MatrixFactorization, CategoryBasedRecommender, CollectivePopularity, \
    MultiVAE, BPRMF, Neural_BPRMF, LightGCN, ItemKNN, PGN
import os
import shutil
import pickle
import numpy as np
import pandas as pd
import torch


def reset_directory(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Delete the directory and all its contents
        shutil.rmtree(path)
    # Recreate the directory
    os.makedirs(path)


log_dir = 'logs/'
for filename in os.listdir(log_dir):
    if filename.endswith('.log'):
        file_path = os.path.join(log_dir, filename)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
print("All .log files deleted successfully!")

# Parameters
threshold = 100  # Performance degradation threshold (adjust as needed)
max_simulation_days = 195  # Maximum number of simulation days
topK = 20
num_nearest_neighbors = 10
num_latent_factors = 32

DEFAULT_TW = 210
DEFAULT_K = 7
DEFAULT_SIM_DAYS = 104
DEFAULT_VERSION = "full"
DEFAULT_SEED = 1
DEFAULT_CITY = "nyc"

parser = argparse.ArgumentParser(description="Parse training parameters")
parser.add_argument("-tw", type=int, default=DEFAULT_TW, help="Training window in days")
parser.add_argument("-k", type=int, default=DEFAULT_K, help="Epoch length in days")
parser.add_argument("-sd", type=int, default=DEFAULT_SIM_DAYS, help="Number of simulation days")
parser.add_argument("-v", type=str, default=DEFAULT_VERSION, help="Version")
parser.add_argument("-s", type=int, default=DEFAULT_SEED, help="Random seed")
parser.add_argument("-c", default=DEFAULT_CITY, help="City")
parser.add_argument("-rb", type=str, required=True, help="p. Required parameter (no default!)")
parser.add_argument("-rs", type=str, required=True, help="RS. Required parameter (no default!)")

# Parsear los argumentos
args = parser.parse_args()
t_window = int(args.tw)
k_days = int(args.k)
version = args.v
seed = int(args.s)
city = args.c
recommender_prob = float(args.rb)
max_simulation_days = int(args.sd)
algorithm = args.rs

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")
# Set the random seed for reproducibility


random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print(f"Using seed: {seed}")

data_file = f"data/processed/foursquare_complete_nyc_cp_{version}.csv"

num_items = 27413
latent_dim = 128
if algorithm == "MF":
    recommender = MatrixFactorization(num_latent_factors=num_latent_factors)
elif algorithm == "CatRec":
    recommender = CategoryBasedRecommender()
elif algorithm == "CPop":
    recommender = CollectivePopularity()
elif algorithm == "UserKNN":
    recommender = UserKNN(k=num_nearest_neighbors)
elif algorithm == "ItemKNN":
    recommender = ItemKNN(k=num_nearest_neighbors)
elif algorithm == "MultiVAE":
    recommender = MultiVAE(input_dim=num_items, latent_dim=latent_dim)
elif algorithm == "BPRMF":
    num_users = 1063
    latent_dim = 128
    recommender = BPRMF(num_users, num_items, latent_dim, epochs=500, lr=0.001,
                        batch_size=16, patience=5, reg_lambda=0.0001, device=device)
    print("BPRMF loaded")
elif algorithm == "Neural_BPRMF":
    num_users = 1063
    latent_dim = 128
    recommender = Neural_BPRMF(num_users, num_items, latent_dim, epochs=500, lr=0.001,
                               batch_size=16, patience=5, reg_lambda=0.0001, device=device)
    print("Neural_BPRMF loaded")
elif algorithm == "LightGCN":
    num_users = 1063
    latent_dim = 128
    recommender = LightGCN(num_users, num_items, latent_dim, num_layers=2, epochs=500, lr=0.001,
                           batch_size=32, patience=5, reg_lambda=0.0001, device=device)
    print("LightGCN loaded")
elif algorithm == "PGN":
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(data_file)
    # Optionally drop duplicates based on 'venueID', keeping the first occurrence.
    df_unique = df.drop_duplicates(subset='venueID')
    # Build the dictionary mapping each venueID to the (lat, lng) tuple.
    venue_coords = {row['venueID']: (row['lat'], row['lng']) for _, row in df_unique.iterrows()}
    recommender = PGN(venue_coords, userknn_k=num_nearest_neighbors)
    print("PGN loaded")

sim = Simulation(city, data_file, t_window, k_days, threshold, max_simulation_days, recommender, topK, recommender_prob,
                 version,
                 category_attribute_name="Second_Category", seed=seed)
sim.run()

experiments_output_path = f'./data/processed/experiments/{version}/{seed}/'
this_experiment_path = experiments_output_path + f"city_{city}__train_{t_window}__step_{k_days}__max_{max_simulation_days}__topK_{topK}__alg_{algorithm}__recProb_{recommender_prob}"
reset_directory(this_experiment_path)

user_histories_path = this_experiment_path + "/user_histories"
reset_directory(user_histories_path)

# get the history of each user and store it into a dictionary
user_histories = {}
for user_id in sim.users:
    user_histories[user_id] = sim.users[user_id].history

# Save the user histories into a csv file
for user_id, history in user_histories.items():
    history.to_csv(user_histories_path + "/" + f'{user_id}_history.csv')

