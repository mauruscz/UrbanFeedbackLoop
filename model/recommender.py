from sklearn.neighbors import NearestNeighbors
import model.processing_utils as processing_utils
import model.itemknn_utils as itemknn_utils
from scipy.sparse import csr_matrix
import numpy as np
import math
import logging
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class Recommender:
    # def return_topK(self, user, topK):
    def return_topK(self, user, topK, venue_category=None, category_to_venues=None):
        raise NotImplementedError()

    def fit(self, dataset):
        raise NotImplementedError()


# -------------------------
# UserKNN Recommender
# -------------------------
class UserKNN(Recommender):
    def __init__(self, k, n_jobs=10):
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k, n_jobs=n_jobs)
        self.k = k
        self.training_dataset = None

    def fit(self, dataset):
        self.training_dataset = dataset
        self.model.fit(dataset)

    def get_scores(self, user, venues_correct_category_at_radius, venue_category=None, category_to_venues=None):
        # Find k-nearest users for the given user.
        distances, indices = self.model.kneighbors(self.training_dataset.loc[[user.uid]], n_neighbors=self.k)
        similar_users = self.training_dataset.index[indices.flatten()]
        # Aggregate the venues visited by similar users.
        similar_users_data = self.training_dataset.loc[similar_users]
        venue_counts = similar_users_data.sum(axis=0)
        # Filter candidate venues.
        valid_venues = set(venues_correct_category_at_radius)
        if venue_category is not None and category_to_venues is not None:
            valid_venues = valid_venues.intersection(set(category_to_venues.get(venue_category, [])))
        # Build a dictionary of scores (higher counts mean higher similarity).
        scores = {venue: venue_counts.loc[venue] for venue in venue_counts.index if venue in valid_venues}
        return scores

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        scores = self.get_scores(user, venues_correct_category_at_radius, venue_category, category_to_venues)
        recommended = processing_utils.sort_and_cut_tuples(list(scores.items()), topK)
        return {venue: score for venue, score in recommended}


# -------------------------
# Pop Recommender
# -------------------------
class Pop(Recommender):
    def __init__(self):
        self.training_dataset = None
        self.venue_counts = None

    def fit(self, dataset):
        self.training_dataset = dataset
        # Pre-compute popularity counts (sum over users for each venue).
        self.venue_counts = dataset.sum(axis=0)

    def get_scores(self, user, venues_correct_category_at_radius, venue_category=None, category_to_venues=None):
        valid_venues = set(venues_correct_category_at_radius)
        if venue_category is not None and category_to_venues is not None:
            valid_venues = valid_venues.intersection(set(category_to_venues.get(venue_category, [])))
        scores = {venue: self.venue_counts.loc[venue] for venue in self.venue_counts.index if venue in valid_venues}
        return scores

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        scores = self.get_scores(user, venues_correct_category_at_radius, venue_category, category_to_venues)
        recommended = processing_utils.sort_and_cut_tuples(list(scores.items()), topK)
        return {venue: score for venue, score in recommended}


# -------------------------
# AvgDis Recommender
# -------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute the Haversine distance (in km) between two latitude-longitude points."""
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class AvgDis(Recommender):
    def __init__(self, venue_coords):
        """
        Parameters:
          - venue_coords: dict mapping each venueID to a tuple (lat, lon)
        """
        self.training_dataset = None
        self.venue_coords = venue_coords
        self.user_avg_locations = {}  # userID -> (avg_lat, avg_lon)
        self.user_venue_distances = {}  # userID -> dict of venue: distance

    def fit(self, dataset):
        self.training_dataset = dataset
        # Pre-compute the average (midpoint) location for each user.
        for user_id, row in dataset.iterrows():
            visited_venues = row[row > 0].index
            lat_list = []
            lon_list = []
            for venue in visited_venues:
                if venue in self.venue_coords:
                    lat, lon = self.venue_coords[venue]
                    lat_list.append(lat)
                    lon_list.append(lon)
            if lat_list:
                avg_lat = np.mean(lat_list)
                avg_lon = np.mean(lon_list)
                self.user_avg_locations[user_id] = (avg_lat, avg_lon)
            else:
                self.user_avg_locations[user_id] = None

        # Pre-compute the distance from each user's average location to every venue.
        for user_id, avg_loc in self.user_avg_locations.items():
            self.user_venue_distances[user_id] = {}
            if avg_loc is not None:
                avg_lat, avg_lon = avg_loc
                for venue, (lat, lon) in self.venue_coords.items():
                    distance = haversine_distance(avg_lat, avg_lon, lat, lon)
                    self.user_venue_distances[user_id][venue] = distance

    def get_scores(self, user, venues_correct_category_at_radius, venue_category=None, category_to_venues=None):
        # Retrieve pre-computed distances for the user.
        if user.uid not in self.user_venue_distances:
            return {}
        distances_dict = self.user_venue_distances[user.uid]
        valid_venues = set(venues_correct_category_at_radius)
        if venue_category is not None and category_to_venues is not None:
            valid_venues = valid_venues.intersection(set(category_to_venues.get(venue_category, [])))
        # Convert distance into a similarity score (higher is better).
        scores = {venue: 1.0 / (1.0 + distances_dict[venue])
                  for venue in valid_venues if venue in distances_dict}
        return scores

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        scores = self.get_scores(user, venues_correct_category_at_radius, venue_category, category_to_venues)
        recommended = processing_utils.sort_and_cut_tuples(list(scores.items()), topK)
        return {venue: score for venue, score in recommended}


class PGN(Recommender):
    def __init__(self, venue_coords, userknn_k=5, n_jobs=10):
        # Instantiate sub-models.
        self.userknn = UserKNN(k=userknn_k, n_jobs=n_jobs)
        self.pop = Pop()
        self.avgdis = AvgDis(venue_coords)

    def fit(self, dataset):
        self.dataset = dataset
        self.userknn.fit(dataset)
        self.pop.fit(dataset)
        self.avgdis.fit(dataset)

    def normalize_scores(self, scores):
        """
        Min-max normalize a dictionary of scores.
        If all scores are equal, return a dictionary with all normalized values set to 1.
        """
        if not scores:
            return {}
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {key: 1.0 for key in scores}
        return {key: (score - min_val) / (max_val - min_val) for key, score in scores.items()}

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        # Get raw scores from each sub-model.
        scores_userknn = self.userknn.get_scores(user, venues_correct_category_at_radius, venue_category,
                                                 category_to_venues)
        scores_pop = self.pop.get_scores(user, venues_correct_category_at_radius, venue_category, category_to_venues)
        scores_avgdis = self.avgdis.get_scores(user, venues_correct_category_at_radius, venue_category,
                                               category_to_venues)

        # Min-max normalize each set of scores.
        norm_scores_userknn = self.normalize_scores(scores_userknn)
        norm_scores_pop = self.normalize_scores(scores_pop)
        norm_scores_avgdis = self.normalize_scores(scores_avgdis)

        # Combine the normalized scores for each venue.
        all_venues = set(norm_scores_userknn.keys()) | set(norm_scores_pop.keys()) | set(norm_scores_avgdis.keys())
        combined_scores = {}
        for venue in all_venues:
            score_u = norm_scores_userknn.get(venue, 0)
            score_p = norm_scores_pop.get(venue, 0)
            score_a = norm_scores_avgdis.get(venue, 0)
            combined_scores[venue] = (score_u + score_p + score_a) / 3.0

        # Sort combined scores in descending order and select the topK venues.
        recommended = processing_utils.sort_and_cut_tuples(list(combined_scores.items()), topK)
        return {venue: score for venue, score in recommended}


class ItemKNN(Recommender):
    def __init__(self, k):
        self.k = k
        self.training_dataset = None

    def fit(self, dataset):
        self.user2id = {userid: i for i, userid in enumerate(dataset.index)}
        self.venue2id = {venueid: i for i, venueid in enumerate(dataset.columns)}
        self.training_dataset = csr_matrix(dataset).astype(np.float32)
        _, self.w = itemknn_utils.ComputeSimilarity(self.training_dataset,
                                                    topk=self.k, shrink=0.0, method='item').compute_similarity()
        self.pred_mat = self.training_dataset.dot(self.w).tolil()

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        recommended_venues = [(venue, self.pred_mat[self.user2id[user.uid], self.venue2id[venue]]) for venue in
                              venues_correct_category_at_radius]
        recommended_venues = processing_utils.sort_and_cut_tuples(recommended_venues, topK)
        return {elem[0]: elem[1] for elem in recommended_venues}


class MatrixFactorization(Recommender):
    def __init__(self, num_latent_factors):
        self.num_latent_factors = num_latent_factors
        self.user_factors = None
        self.item_factors = None
        self.user_index = {}
        self.item_index = {}

    def fit(self, dataset):
        # Convert the dataset into a matrix with rows as users and columns as items
        matrix = dataset.fillna(0).values

        # Map user and item IDs to matrix indices
        self.user_index = {user: idx for idx, user in enumerate(dataset.index)}
        self.item_index = {item: idx for idx, item in enumerate(dataset.columns)}

        # Apply Singular Value Decomposition
        # Keep only the top factors which explain most of the variance
        U, sigma, VT = np.linalg.svd(matrix, full_matrices=False)

        # Convert sigma into a diagonal matrix
        sigma = np.diag(sigma)

        # Reduce dimensions based on sigma's size or a predefined k (number of latent factors)
        k = min(self.num_latent_factors,
                sigma.shape[0])  # Let's assume we use 20 factors or as many as we have if fewer
        self.user_factors = np.dot(U[:, :k], sigma[:k, :k])
        self.item_factors = VT[:k, :].T

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        # Get user index
        user_idx = self.user_index[user.uid]

        # Convert to set early if not already (for O(1) lookups)
        correct = set(venues_correct_category_at_radius)

        # Filter item_index to only include relevant venues BEFORE dot product
        relevant_items = {item: idx for item, idx in self.item_index.items() if item in correct}

        # Only compute scores for relevant items
        relevant_scores = []
        user_factor = self.user_factors[user_idx]

        for item, idx in relevant_items.items():
            # Calculate score only for this item
            score = np.dot(user_factor, self.item_factors[idx])
            relevant_scores.append((item, score))

        # Sort and cut to topK
        recommended_venues = processing_utils.sort_and_cut_tuples(relevant_scores, topK)

        return {elem[0]: elem[1] for elem in recommended_venues}


class CollectivePopularity(Recommender):
    def __init__(self):
        self.venue_ids = None
        self.relative_popularity = None

    def fit(self, dataset):
        # Convert the dataset into a matrix with rows as users and columns as items
        dataset = dataset.fillna(0)

        # Compute popularity
        popularity = dataset.sum() / len(dataset)

        # Sort by descending popularity
        sorted_popularity = popularity.sort_values(ascending=False)

        # Extract venue IDs and relative popularity
        self.venue_ids = np.array(sorted_popularity.index.tolist())
        self.relative_popularity = np.array(sorted_popularity.values.tolist())

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        # First filter by venues within the correct radius
        mask = np.isin(self.venue_ids, venues_correct_category_at_radius)

        # If venue_category is provided, also filter by category
        if venue_category is not None and category_to_venues is not None:
            category_mask = np.isin(self.venue_ids, category_to_venues[venue_category])
            # Combine both masks - venue must be in both correct radius AND correct category
            mask = np.logical_and(mask, category_mask)

        if mask.sum() == 0:
            print('CollectivePopularity cannot recommend anything')
            return {}

        # Filter venue IDs and their popularities using the mask
        filtered_venue_ids = self.venue_ids[mask]
        filtered_popularity = self.relative_popularity[mask]

        # Sort by popularity (should already be sorted, but ensuring it)
        sorted_indices = np.argsort(filtered_popularity)[::-1]  # Descending order

        # Get top K venues
        top_k_indices = sorted_indices[:min(topK, len(sorted_indices))]

        # Return dictionary of venue IDs and their popularity scores
        return {filtered_venue_ids[i]: filtered_popularity[i] for i in top_k_indices}


class CategoryBasedRecommender(Recommender):
    def __init__(self):
        self.item_popularity = {}

    def fit(self, dataset):
        """
        dataset: pandas DataFrame (users x items), values are interactions (visit counts)
        """
        # Compute overall popularity per venue
        self.item_popularity = dataset.sum(axis=0).to_dict()

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        """
        user: User object having 'uid' and 'history' (DataFrame of user's visits)
        venues_correct_category_at_radius: already filtered list of candidate venue IDs at radius and in the desired category
        """
        # Extract venue IDs from user history DataFrame
        if isinstance(user.history, pd.DataFrame):
            user_visited_venues = user.history['venueID'].tolist()
        else:
            user_visited_venues = user.history  # Fallback if it's already a list

        # Count occurrences of each venue in the user's history
        user_history_counts = Counter(user_visited_venues)
        # Check if the user has visited any of the venues in venues_correct_category_at_radius
        visited_candidate_venues = [venue for venue in venues_correct_category_at_radius
                                    if user_history_counts.get(venue, 0) > 0]
        has_visited_any = len(visited_candidate_venues) > 0

        if has_visited_any:
            # If user has visited any venues in the candidate list, score based on visit count
            venue_scores = [
                (venue, user_history_counts.get(venue, 0))
                for venue in venues_correct_category_at_radius
            ]
        else:
            # Log the fallback
            if hasattr(user, 'simulation') and hasattr(user.simulation, 'step_counter'):
                logging.info(f'User {user.uid} has no history in venues_correct_category_at_radius. '
                             f'Fallback to random selection. Step {user.simulation.step_counter}')
                # logging.info(f'User history: {user.history}')
                # logging.info("------")

            # If user hasn't visited any venues in the candidate list, assign equal scores (random)
            venue_scores = [(venue, 1) for venue in venues_correct_category_at_radius]

        # Apply sort_and_cut_tuples to get the top K venues
        recommended_venues = processing_utils.sort_and_cut_tuples(venue_scores, topK)

        # If using history-based scoring, zero-visit venues would be at the bottom
        # after sorting and might be cut off if there are enough visited venues
        return {venue: score for venue, score in recommended_venues}


# Base class placeholder (assume it is defined elsewhere)
class Recommender:
    pass


# VAE network definition for collaborative filtering
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.5):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # Encoder: input -> hidden -> latent parameters
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> hidden -> output (reconstruction)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.dropout(h3)
        # Using sigmoid to bound the reconstruction between 0 and 1.
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # Note: During training, we use the reparameterization trick.
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# Loss function combining reconstruction loss and KL divergence.
def loss_function(recon_x, x, mu, logvar):
    # Using binary cross-entropy as reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# MultiVAE recommender class with early stopping
class MultiVAE(Recommender):
    def __init__(self, input_dim, latent_dim, dropout=0.5, epochs=50, lr=0.001, patience=5, device='cpu'):
        """
        Parameters:
          hidden_dim: number of neurons in the hidden layer
          latent_dim: number of latent factors
          dropout: dropout rate for the encoder and decoder
          epochs: maximum number of training epochs
          lr: learning rate
          patience: number of epochs to wait for improvement before stopping
          device: 'cpu' or 'cuda'
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.user_index = {}
        self.item_index = {}
        self.user_item_matrix = None  # To store the training data
        # Initialize the VAE model and optimizer
        self.model = VAE(self.input_dim, 1024, self.latent_dim, self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, dataset):
        """
        dataset: pandas DataFrame with rows as users and columns as items.
        Assumes entries are binary or scaled between 0 and 1.
        """
        # Map users and items to indices
        self.user_index = {user: idx for idx, user in enumerate(dataset.index)}
        self.item_index = {item: idx for idx, item in enumerate(dataset.columns)}

        # Fill missing values and convert the DataFrame to a numpy array.
        data_matrix = dataset.fillna(0).values.astype(np.float32)
        data_matrix[data_matrix > 0] = 1.0  # MultiVAE cannot accept values > 0

        self.user_item_matrix = torch.FloatTensor(data_matrix).to(self.device)

        self.model.train()

        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(self.user_item_matrix)
            loss = loss_function(recon_batch, self.user_item_matrix, mu, logvar)
            loss.backward()
            self.optimizer.step()

            # Check for improvement
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Optionally print loss information
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.2f}")

            # Early stopping condition
            if epochs_without_improvement >= self.patience:
                # Optionally print early stopping message
                # print(f"Stopping early at epoch {epoch+1} with best loss {best_loss:.2f}")
                break

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        """
        Returns the topK recommended items for a given user.
        Parameters:
          user: an object with an attribute uid (unique user id)
          venues_correct_category_at_radius: list or set of item ids to consider for recommendation
          topK: number of recommendations to return
        """
        # Retrieve the user's row index
        user_idx = self.user_index[user.uid]

        # Filter to only the relevant items (venues)
        correct_items = set(venues_correct_category_at_radius)
        relevant_item_indices = {item: idx for item, idx in self.item_index.items() if item in correct_items}

        self.model.eval()
        with torch.no_grad():
            # Get the user's input vector
            user_vector = self.user_item_matrix[user_idx].unsqueeze(0)  # shape: [1, num_items]
            # During inference, bypass reparameterization by directly using the mean.
            mu, _ = self.model.encode(user_vector)
            recon = self.model.decode(mu)
            scores = recon.squeeze(0).cpu().numpy()  # predicted probabilities for each item

        # Filter scores for only relevant items
        relevant_scores = []
        for item, idx in relevant_item_indices.items():
            score = scores[idx]
            relevant_scores.append((item, score))

        # Sort items based on score and take the topK
        recommended = sorted(relevant_scores, key=lambda x: x[1], reverse=True)[:topK]
        # Return a dictionary mapping item id to its predicted score
        return {item: score for item, score in recommended}


def bpr_loss(user_factors, pos_item_factors, neg_item_factors, reg_lambda=0.0):
    """
    Computes the BPR loss given user, positive, and negative item factors.
    - user_factors: tensor of shape [batch_size, latent_dim]
    - pos_item_factors: tensor of shape [batch_size, latent_dim]
    - neg_item_factors: tensor of shape [batch_size, latent_dim]
    - reg_lambda: regularization coefficient for L2 regularization
    """
    diff = (user_factors * (pos_item_factors - neg_item_factors)).sum(dim=1)
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()  # epsilon to avoid log(0)

    if reg_lambda > 0:
        reg_loss = reg_lambda * (user_factors.norm(2).pow(2) +
                                 pos_item_factors.norm(2).pow(2) +
                                 neg_item_factors.norm(2).pow(2)) / user_factors.shape[0]
        loss += reg_loss
    return loss


class BPRMF(Recommender):
    def __init__(self, num_users, num_items, latent_dim, epochs=50, lr=0.001,
                 batch_size=128, patience=5, reg_lambda=0.0, device='cpu'):
        """
        Parameters:
          num_users: total number of users.
          num_items: total number of items.
          latent_dim: number of latent factors for both users and items.
          epochs: maximum number of training epochs.
          lr: learning rate.
          batch_size: mini-batch size for training.
          patience: number of epochs to wait for improvement before stopping.
          reg_lambda: regularization coefficient (L2 penalty) on latent factors.
          device: 'cpu' or 'cuda'.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.reg_lambda = reg_lambda
        self.device = torch.device(device)

        # Mappings will be built from the dataset indices.
        self.user_index = {}
        self.item_index = {}
        self.user_item_matrix = None  # binary interaction matrix (as numpy array)

        # Precomputed negative sampling weights dictionary:
        # key: user index, value: np.array of shape (num_items,)
        self.user_neg_weights = {}

        self.user_factors = nn.Parameter(torch.empty(self.num_users, self.latent_dim, device=self.device))
        self.item_factors = nn.Parameter(torch.empty(self.num_items, self.latent_dim, device=self.device))
        nn.init.normal_(self.user_factors, std=0.01)
        nn.init.normal_(self.item_factors, std=0.01)

        self.optimizer = torch.optim.Adam([self.user_factors, self.item_factors], lr=self.lr)

    def fit(self, dataset):
        """
        dataset: pandas DataFrame with rows as users and columns as items.
                 The index/columns should match the order used to initialize the model.
                 Non-zero entries indicate positive interactions.
        """
        # Build mappings based on the dataset.
        self.user_index = {user: idx for idx, user in enumerate(dataset.index)}
        self.item_index = {item: idx for idx, item in enumerate(dataset.columns)}

        # Convert dataset to a binary numpy array.
        data_matrix = dataset.fillna(0).values.astype(np.float32)
        data_matrix[data_matrix > 0] = 1.0  # ensure binary interactions
        self.user_item_matrix = data_matrix

        # Precompute positive items per user and weighted negative sampling distributions.
        user_pos_items = {}
        for u in range(self.num_users):
            pos_items = np.where(self.user_item_matrix[u] > 0)[0]
            user_pos_items[u] = pos_items

            # Create weights: default weight 1 for all items.
            weights = np.ones(self.num_items, dtype=np.float32)
            weights[pos_items] = 0.0  # zero out positive items.
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(self.num_items, dtype=np.float32) / self.num_items
            self.user_neg_weights[u] = weights

        best_loss = float('inf')
        epochs_without_improvement = 0

        num_triplets_per_user = 4  # generate 4 triplets per user.

        for epoch in range(self.epochs):
            triplet_users = []
            triplet_pos = []
            triplet_neg = []

            # Generate triplets for all users.
            for u in range(self.num_users):
                pos_items = user_pos_items[u]
                if len(pos_items) == 0:
                    continue  # skip users with no positive interactions.
                for _ in range(num_triplets_per_user):
                    pos_item = np.random.choice(pos_items)
                    neg_item = np.random.choice(np.arange(self.num_items), p=self.user_neg_weights[u])
                    triplet_users.append(u)
                    triplet_pos.append(pos_item)
                    triplet_neg.append(neg_item)

            if len(triplet_users) == 0:
                break  # no training samples available.

            # Shuffle triplets.
            triplets = list(zip(triplet_users, triplet_pos, triplet_neg))
            np.random.shuffle(triplets)
            triplet_users, triplet_pos, triplet_neg = zip(*triplets)
            triplet_users = np.array(triplet_users)
            triplet_pos = np.array(triplet_pos)
            triplet_neg = np.array(triplet_neg)

            epoch_loss = 0.0
            num_batches = int(np.ceil(len(triplet_users) / self.batch_size))

            # Process mini-batches.
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, len(triplet_users))
                batch_users = torch.LongTensor(triplet_users[start:end]).to(self.device)
                batch_pos = torch.LongTensor(triplet_pos[start:end]).to(self.device)
                batch_neg = torch.LongTensor(triplet_neg[start:end]).to(self.device)

                u_batch = self.user_factors[batch_users]  # shape: [batch_size, latent_dim]
                pos_batch = self.item_factors[batch_pos]  # shape: [batch_size, latent_dim]
                neg_batch = self.item_factors[batch_neg]  # shape: [batch_size, latent_dim]

                loss = bpr_loss(u_batch, pos_batch, neg_batch, self.reg_lambda)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Average loss per epoch.
            avg_epoch_loss = epoch_loss / num_batches

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Optionally print training loss.
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}")

            if epochs_without_improvement >= self.patience:
                # print(f"Stopping early at epoch {epoch+1} with best loss {best_loss:.4f}")
                break

    def return_topK(self, user, venues_correct_category_at_radius, topK,
                    venue_category=None, category_to_venues=None):
        """
        Returns the topK recommended items for a given user.
        Parameters:
          user: an object with an attribute uid (unique user id).
          venues_correct_category_at_radius: list or set of item ids to consider for recommendation.
          topK: number of recommendations to return.
        """
        user_idx = self.user_index[user.uid]

        # Switch to evaluation mode by disabling gradient computation.
        with torch.no_grad():
            u_vector = self.user_factors[user_idx].unsqueeze(0)  # shape: [1, latent_dim]
            all_item_vectors = self.item_factors  # shape: [num_items, latent_dim]
            scores = torch.matmul(u_vector, all_item_vectors.t()).squeeze(0).cpu().numpy()

        correct_items = set(venues_correct_category_at_radius)
        relevant_scores = [(item, scores[idx])
                           for item, idx in self.item_index.items()
                           if item in correct_items]

        recommended = sorted(relevant_scores, key=lambda x: x[1], reverse=True)[:topK]
        return {item: score for item, score in recommended}


class Neural_BPRMF(Recommender):
    def __init__(self, num_users, num_items, latent_dim, epochs=50, lr=0.001,
                 batch_size=128, patience=5, reg_lambda=0.0, device='cpu'):
        """
        Parameters:
          num_users: total number of users.
          num_items: total number of items.
          latent_dim: number of latent factors for both users and items.
          epochs: maximum number of training epochs.
          lr: learning rate.
          batch_size: mini-batch size for training.
          patience: number of epochs to wait for improvement before stopping.
          reg_lambda: regularization coefficient (L2 penalty) on latent factors.
          device: 'cpu' or 'cuda'.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.reg_lambda = reg_lambda
        self.device = torch.device(device)

        # Mappings will be built from the dataset indices.
        self.user_index = {}
        self.item_index = {}
        self.user_item_matrix = None  # binary interaction matrix (as numpy array)

        # Precomputed negative sampling weights dictionary:
        # key: user index, value: np.array of shape (num_items,)
        self.user_neg_weights = {}

        # Initialize embeddings and optimizer once.
        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        self.user_embedding.to(self.device)
        self.item_embedding.to(self.device)
        self.optimizer = torch.optim.Adam(list(self.user_embedding.parameters()) +
                                          list(self.item_embedding.parameters()), lr=self.lr)

    def fit(self, dataset):
        """
        dataset: pandas DataFrame with rows as users and columns as items.
                 The index/columns should match the order used to initialize the model.
                 Non-zero entries indicate positive interactions.
        """
        # Build mappings based on the dataset.
        self.user_index = {user: idx for idx, user in enumerate(dataset.index)}
        self.item_index = {item: idx for idx, item in enumerate(dataset.columns)}

        # Convert dataset to a binary numpy array.
        data_matrix = dataset.fillna(0).values.astype(np.float32)
        data_matrix[data_matrix > 0] = 1.0  # ensure binary interactions
        self.user_item_matrix = data_matrix

        # Precompute positive items per user and weighted negative sampling distributions.
        user_pos_items = {}
        for u in range(self.num_users):
            pos_items = np.where(self.user_item_matrix[u] > 0)[0]
            user_pos_items[u] = pos_items

            # Create weights: default weight 1 for all items.
            weights = np.ones(self.num_items, dtype=np.float32)
            weights[pos_items] = 0.0  # zero out positive items.
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(self.num_items, dtype=np.float32) / self.num_items
            self.user_neg_weights[u] = weights

        best_loss = float('inf')
        epochs_without_improvement = 0

        self.user_embedding.train()
        self.item_embedding.train()

        num_triplets_per_user = 4  # generate 4 triplets per user.

        for epoch in range(self.epochs):
            triplet_users = []
            triplet_pos = []
            triplet_neg = []

            # Generate triplets for all users.
            for u in range(self.num_users):
                pos_items = user_pos_items[u]
                if len(pos_items) == 0:
                    continue  # skip users with no positive interactions.
                for _ in range(num_triplets_per_user):
                    pos_item = np.random.choice(pos_items)
                    neg_item = np.random.choice(np.arange(self.num_items), p=self.user_neg_weights[u])
                    triplet_users.append(u)
                    triplet_pos.append(pos_item)
                    triplet_neg.append(neg_item)

            if len(triplet_users) == 0:
                break  # no training samples available.

            # Shuffle triplets.
            triplets = list(zip(triplet_users, triplet_pos, triplet_neg))
            np.random.shuffle(triplets)
            triplet_users, triplet_pos, triplet_neg = zip(*triplets)
            triplet_users = np.array(triplet_users)
            triplet_pos = np.array(triplet_pos)
            triplet_neg = np.array(triplet_neg)

            epoch_loss = 0.0
            num_batches = int(np.ceil(len(triplet_users) / self.batch_size))

            # Process mini-batches.
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, len(triplet_users))
                batch_users = torch.LongTensor(triplet_users[start:end]).to(self.device)
                batch_pos = torch.LongTensor(triplet_pos[start:end]).to(self.device)
                batch_neg = torch.LongTensor(triplet_neg[start:end]).to(self.device)

                # Lookup embeddings.
                u_factors = self.user_embedding(batch_users)
                pos_factors = self.item_embedding(batch_pos)
                neg_factors = self.item_embedding(batch_neg)

                loss = bpr_loss(u_factors, pos_factors, neg_factors, self.reg_lambda)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Average loss per epoch.
            avg_epoch_loss = epoch_loss / num_batches

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Optionally print training loss.
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}")

            if epochs_without_improvement >= self.patience:
                # print(f"Stopping early at epoch {epoch+1} with best loss {best_loss:.4f}")
                break

    def return_topK(self, user, venues_correct_category_at_radius, topK,
                    venue_category=None, category_to_venues=None):
        """
        Returns the topK recommended items for a given user.
        Parameters:
          user: an object with an attribute uid (unique user id).
          venues_correct_category_at_radius: list or set of item ids to consider for recommendation.
          topK: number of recommendations to return.
        """
        user_idx = self.user_index[user.uid]

        self.user_embedding.eval()
        self.item_embedding.eval()

        with torch.no_grad():
            u_factor = self.user_embedding(torch.LongTensor([user_idx]).to(self.device))
            all_item_factors = self.item_embedding.weight  # shape: [num_items, latent_dim]
            scores = torch.matmul(u_factor, all_item_factors.t()).squeeze(0).cpu().numpy()

        correct_items = set(venues_correct_category_at_radius)
        relevant_scores = [(item, scores[idx])
                           for item, idx in self.item_index.items()
                           if item in correct_items]

        recommended = sorted(relevant_scores, key=lambda x: x[1], reverse=True)[:topK]
        return {item: score for item, score in recommended}


class LightGCN(Recommender):
    def __init__(self, num_users, num_items, latent_dim, num_layers=3, epochs=50, lr=0.001,
                 batch_size=128, patience=5, reg_lambda=0.0, device='cpu'):
        """
        LightGCN implementation for recommendation using graph convolution on the user-item bipartite graph.
        Parameters:
          num_users: total number of users.
          num_items: total number of items.
          latent_dim: dimension of latent embeddings.
          num_layers: number of graph convolutional layers.
          epochs: maximum number of training epochs.
          lr: learning rate.
          batch_size: mini-batch size.
          patience: epochs to wait for improvement before early stopping.
          reg_lambda: L2 regularization coefficient.
          device: 'cpu' or 'cuda'.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.reg_lambda = reg_lambda
        self.device = torch.device(device)

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        self.user_embedding.to(self.device)
        self.item_embedding.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.user_embedding.parameters()) +
                                          list(self.item_embedding.parameters()), lr=self.lr)

        self.user_index = {}
        self.item_index = {}
        self.user_item_matrix = None  # binary interaction matrix (numpy array)
        self.A_hat = None  # normalized adjacency matrix (sparse tensor)
        self.user_pos_items = {}  # positive items per user for sampling
        self.user_neg_weights = {}  # negative sampling weights per user

    def build_graph(self):
        """
        Build the normalized bipartite graph from the user-item interaction matrix.
        The graph is represented as a sparse tensor A_hat.
        """
        import scipy.sparse as sp
        num_users = self.num_users
        num_items = self.num_items
        user_item = self.user_item_matrix  # shape: (num_users, num_items)
        # Get indices of positive interactions
        user_indices, item_indices = np.nonzero(user_item)
        # For each interaction, add edges (u, num_users+i) and (num_users+i, u)
        rows = np.concatenate([user_indices, item_indices + num_users])
        cols = np.concatenate([item_indices + num_users, user_indices])
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))

        # Normalize A using D^{-1/2} A D^{-1/2}
        rowsum = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_normalized = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()

        indices = torch.from_numpy(np.vstack((A_normalized.row, A_normalized.col)).astype(np.int64))
        values = torch.from_numpy(A_normalized.data.astype(np.float32))
        shape = A_normalized.shape
        self.A_hat = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(self.device)

    def compute_final_embeddings(self):
        """
        Perform graph convolution to compute final user and item embeddings.
        Returns:
          final_user_embeddings: tensor of shape [num_users, latent_dim]
          final_item_embeddings: tensor of shape [num_items, latent_dim]
        """
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        for layer in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.A_hat, all_embeddings)
            embeddings_list.append(all_embeddings)
        final_embeddings = sum(embeddings_list) / (self.num_layers + 1)
        final_user_embeddings = final_embeddings[:self.num_users]
        final_item_embeddings = final_embeddings[self.num_users:]
        return final_user_embeddings, final_item_embeddings

    def fit(self, dataset):
        """
        Train the LightGCN model.
        Parameters:
          dataset: pandas DataFrame with rows as users and columns as items.
                   Non-zero entries indicate positive interactions.
        """
        # Build user and item index mappings
        self.user_index = {user: idx for idx, user in enumerate(dataset.index)}
        self.item_index = {item: idx for idx, item in enumerate(dataset.columns)}
        data_matrix = dataset.fillna(0).values.astype(np.float32)
        data_matrix[data_matrix > 0] = 1.0
        self.user_item_matrix = data_matrix
        self.build_graph()

        # Precompute positive items and negative sampling weights for each user
        for u in range(self.num_users):
            pos_items = np.where(self.user_item_matrix[u] > 0)[0]
            self.user_pos_items[u] = pos_items
            weights = np.ones(self.num_items, dtype=np.float32)
            weights[pos_items] = 0.0
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(self.num_items, dtype=np.float32) / self.num_items
            self.user_neg_weights[u] = weights

        best_loss = float('inf')
        epochs_without_improvement = 0
        num_triplets_per_user = 4

        for epoch in range(self.epochs):
            triplet_users, triplet_pos, triplet_neg = [], [], []
            for u in range(self.num_users):
                pos_items = self.user_pos_items[u]
                if len(pos_items) == 0:
                    continue
                for _ in range(num_triplets_per_user):
                    pos_item = np.random.choice(pos_items)
                    neg_item = np.random.choice(np.arange(self.num_items), p=self.user_neg_weights[u])
                    triplet_users.append(u)
                    triplet_pos.append(pos_item)
                    triplet_neg.append(neg_item)
            if len(triplet_users) == 0:
                break
            # Shuffle triplets
            triplets = list(zip(triplet_users, triplet_pos, triplet_neg))
            np.random.shuffle(triplets)
            triplet_users, triplet_pos, triplet_neg = zip(*triplets)
            triplet_users = np.array(triplet_users)
            triplet_pos = np.array(triplet_pos)
            triplet_neg = np.array(triplet_neg)

            epoch_loss = 0.0
            num_batches = int(np.ceil(len(triplet_users) / self.batch_size))
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, len(triplet_users))
                batch_users = torch.LongTensor(triplet_users[start:end]).to(self.device)
                batch_pos = torch.LongTensor(triplet_pos[start:end]).to(self.device)
                batch_neg = torch.LongTensor(triplet_neg[start:end]).to(self.device)

                # Compute final embeddings for current parameters
                final_user_embeddings, final_item_embeddings = self.compute_final_embeddings()
                u_emb = final_user_embeddings[batch_users]
                pos_emb = final_item_embeddings[batch_pos]
                neg_emb = final_item_embeddings[batch_neg]

                loss = bpr_loss(u_emb, pos_emb, neg_emb, self.reg_lambda)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / num_batches
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break

    def return_topK(self, user, venues_correct_category_at_radius, topK, venue_category=None, category_to_venues=None):
        """
        Return the topK recommended items for a given user.
        Parameters:
          user: an object with attribute uid.
          venues_correct_category_at_radius: list or set of candidate item ids.
          topK: number of recommendations to return.
        """
        with torch.no_grad():
            final_user_embeddings, final_item_embeddings = self.compute_final_embeddings()
            user_idx = self.user_index[user.uid]
            u_emb = final_user_embeddings[user_idx].unsqueeze(0)  # shape: [1, latent_dim]
            scores = torch.matmul(u_emb, final_item_embeddings.t()).squeeze(0).cpu().numpy()
        correct_items = set(venues_correct_category_at_radius)
        relevant_scores = [(item, scores[idx])
                           for item, idx in self.item_index.items() if item in correct_items]
        recommended = sorted(relevant_scores, key=lambda x: x[1], reverse=True)[:topK]
        return {item: score for item, score in recommended}
