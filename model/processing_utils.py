import numpy as np
from geopy.distance import geodesic
from scipy.spatial import KDTree
import geopandas as gpd
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform

from tqdm import tqdm
import os


def sort_and_cut_tuples(input_list, k):
    """
    Sorts a list of tuples by the scalar value in descending order and
    returns the top k tuples.

    Parameters:
    - input_list (list of tuples): Each tuple contains a string and a scalar value.
    - k (int): The number of top elements to retain.

    Returns:
    - list of tuples: Sorted and truncated list of tuples.
    """
    # Sort the list by the scalar value (second element of the tuple) in descending order
    sorted_list = sorted(input_list, key=lambda x: x[1], reverse=True)
    # Return the top k elements
    return sorted_list[:k]


def sample_weighted(recommendation_list, random_state):
    # Extract keys and values from the dictionary
    keys = list(recommendation_list.keys())
    values = np.array(list(recommendation_list.values()))
    # Normalize the values to create a probability distribution
    probabilities = values / values.sum()
    # Sample one key based on the probabilities
    sampled_index = random_state.choice(len(keys), p=probabilities)
    return keys[sampled_index]


# Compute distances between consecutive rows for the same user
def calculate_distance(row1, row2):
    coords_1 = (row1['lat'], row1['lng'])
    coords_2 = (row2['lat'], row2['lng'])
    return geodesic(coords_1, coords_2).meters


# Haversine formula for vectorized distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def haversine_metric(coords1, coords2):
    R = 6371000  # Earth's radius in meters
    lat1, lon1 = np.radians(coords1[0]), np.radians(coords1[1])
    lat2, lon2 = np.radians(coords2[0]), np.radians(coords2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def calculate_distance_matrix(training_data, city, data_version, trainWindow):
    distance_matrix_fname = f'data/processed/distance_matrix_{city}_{data_version}_trainWindow_{trainWindow}.csv'
    print(distance_matrix_fname)

    saved = os.path.exists(distance_matrix_fname)

    if saved:
        print("Loading distance matrix from file")
        distance_matrix = pd.read_csv(distance_matrix_fname, index_col=0)
        print(distance_matrix.head())
        distance_matrix.index = list(map(lambda x: str(x), distance_matrix.index))
        distance_matrix.columns = list(map(lambda x: str(x), distance_matrix.columns))
        return distance_matrix

    print("Calculating distance matrix")
    # Step 1: Extract unique POIs
    unique_pois = training_data.drop_duplicates(subset=['venueID'])[['lat', 'lng', 'venueID']].set_index('venueID')

    # Step 2: Compute vectorized haversine distance
    coords = unique_pois[['lat', 'lng']].to_numpy()
    distance_matrix = squareform(pdist(coords, metric=haversine_metric))

    # Convert to DataFrame
    distance_matrix_df = pd.DataFrame(distance_matrix, index=unique_pois.index, columns=unique_pois.index)

    # Save result
    distance_matrix_df.to_csv(distance_matrix_fname)
    return distance_matrix_df



def calculate_radius_df(training_data, radius_mt, city, category_column, version, trainWindow):

    radius_path = f'data/processed/radius_df_{city}_radius_{radius_mt}_version_{version}_trainWindow_{trainWindow}.csv'
    if os.path.exists(radius_path):
        print("Loading radius df from file")
        radius_df = pd.read_csv(radius_path, index_col=0)
        radius_df['venueID'] = radius_df['venueID'].astype(str)
        return radius_df


    df = training_data.copy()

    # Get unique venues
    #unique_venues = df.drop_duplicates(['venueID', 'lat', 'lng', category_column])
    unique_venues = df.drop_duplicates(['venueID'])
    
    # Create arrays for vectorized calculation
    lat1 = unique_venues['lat'].values[:, np.newaxis]
    lon1 = unique_venues['lng'].values[:, np.newaxis]
    lat2 = unique_venues['lat'].values
    lon2 = unique_venues['lng'].values
    
    # Calculate distances between all pairs
    distances = haversine(lat1, lon1, lat2, lon2)
    
    # Create category comparison matrix
    categories = unique_venues[category_column].values
    same_category = (categories[:, np.newaxis] == categories)
    
    # Create distance mask
    distance_mask = (distances <= radius_mt) & (distances > 0)
    
    # Count all venues within radius (excluding self)
    total_nearby = distance_mask.sum(axis=1)
    
    # Count venues of same category within radius (excluding self)
    category_mask = distance_mask & same_category
    nearby_same_category = category_mask.sum(axis=1)
    
    radius_df =  pd.DataFrame({
        'venueID': unique_venues['venueID'],
        'category': unique_venues[category_column],
        'nearby_same_category': nearby_same_category,
        'total_nearby': total_nearby
    })

    radius_df.to_csv(radius_path)
    return radius_df





def get_nearby_venues(lat, lng, radius, training_df):
    # Calculate distances
    distances = haversine(lat, lng, training_df['lat'], training_df['lng'])
    
    # Create a mask for venues within radius
    mask = (distances <= radius) & (distances > 0)
    
    # Return list of nearby venues
    return training_df[mask]["venueID"].unique().tolist()



def select_weighted_venues_from_radius(venue_ids, df_radius, column = "nearby_same_category"):

    
    # Get counts for each venue
    counts = df_radius[df_radius["venueID"].isin(venue_ids)][column]
    
    
    # Calculate probabilities
    total_counts = counts.sum()
    if total_counts == 0:
        # If all counts are zero, assign uniform probabilities
        probs = np.ones(len(venue_ids)) / len(venue_ids)
        if len(venue_ids) == 0:
            print("No venues at radius")
            #return a random venue from df_radius
            return df_radius.sample().iloc[0]["venueID"]
    else:
        probs = counts / total_counts
    
    # Randomly select venue
    return np.random.choice(venue_ids, p=probs)


def calculate_distance_list(training_data):
    distance_list = []

    # Process each user's visits
    for uid, user_visits in training_data.groupby('uid'):

        visited_venues = set()   # to keep track of venues already visited
        last_lat, last_lng = None, None  # coordinates of the last new (not repeated) POI
        
        # It's important that the visits are in chronological order.
        # Here we assume that the DataFrame's order represents the visit order.
        for idx, row in user_visits.iterrows():
            venue = row['venueID']
            
            # Check if the venue is being visited for the first time
            if venue not in visited_venues:
                # If this is not the first new venue, calculate the distance from the last one
                if last_lat is not None and last_lng is not None:
                    d = haversine(last_lat, last_lng, row['lat'], row['lng'])
                    distance_list.append(d)
                
                # Mark the venue as visited and update last visited coordinates
                visited_venues.add(venue)
                last_lat, last_lng = row['lat'], row['lng']
            else:
                last_lat, last_lng = row['lat'], row['lng']
                # If the venue was already visited, we skip calculating the distance
                continue

    return distance_list


def pick_radius_weighted(distance_list, random_state):
    # sort the distances
    distance_list.sort()

    n = len(distance_list)

    ecdf_values = np.arange(1, n + 1) / n

    # Pick a random number between 0 and 1
    u = random_state.rand()

    # Find the index of the first distance that is greater than the random number
    idx = np.searchsorted(ecdf_values, u)

    # Return the corresponding distance
    return distance_list[idx]


def extract_word(filename):
    # Remove .csv extension
    base = filename.replace('.csv', '')
    # Split by underscore
    parts = base.split('_')
    # Return last part if it's small/medium, otherwise empty string
    last = parts[-1]
    return last if last in ['small', 'medium'] else ''



def calculate_kd_tree(train_data):
    """
    Builds and returns a KDTree from venue coordinates in train_data.
    
    Parameters:
        train_data (pd.DataFrame): DataFrame containing 'venueID', 'lat', 'lng'.
    
    Returns:
        kd_tree (scipy.spatial.KDTree): KD-Tree for fast nearest neighbor lookup.
        venue_ids (np.array): Array mapping index positions to venue IDs.
    """
    # Extract unique venues
    unique_pois = train_data.drop_duplicates(subset=['venueID'])[['lat', 'lng', 'venueID']].set_index('venueID')

    # Convert to NumPy arrays for KDTree
    coords = unique_pois[['lat', 'lng']].to_numpy()
    venue_ids = unique_pois.index.to_numpy()

    # Build KDTree
    kd_tree = KDTree(coords)
    
    return kd_tree, venue_ids  # Return both tree and venue ID mapping



def find_nearest_venue(kd_tree, venue_ids, venues_of_category, current_venue_id, user_history=set(), 
                       initial_k=50, expand_k=50):
    """
    Finds the nearest venue from `venues_of_category` using a KDTree, avoiding venues in the user's history.

    Parameters:
        kd_tree (scipy.spatial.KDTree): The precomputed KDTree for fast lookups.
        venue_ids (np.array): Mapping from index positions to venue IDs.
        venues_of_category (set): A set of venueIDs that are valid choices.
        current_venue_id (str): The venue ID of the current location.
        user_history (set): A set of venueIDs that the user has already visited.
        initial_k (int): Initial number of neighbors to consider.
        expand_k (int): Step size for expanding the search.

    Returns:
        str: The venueID of the nearest venue, or None if not found.
    """
    # Validate venue existence
    if current_venue_id not in venue_ids:
        return None  # Venue not found in training data

    current_idx = np.where(venue_ids == current_venue_id)[0][0]  # Get index in KDTree
    current_coord = kd_tree.data[current_idx]  # Get the coordinate

    # Exclude visited venues
    valid_venues = venues_of_category - user_history

    k = initial_k  # Start with initial k
    while True:
        # Query KDTree for k-nearest neighbors
        distances, indices = kd_tree.query(current_coord, k=k)

        # Filter results by `venues_of_category` and `user_history`
        for idx, dist in zip(indices, distances):
            neighbor_id = venue_ids[idx]
            if neighbor_id in valid_venues:
                return neighbor_id  # Return the first matching venue
        
        # If no valid venue found, expand search
        k += expand_k
        if k > len(venue_ids):  # Stop if we've checked all venues
            break

    return None  # No valid venue found within the dataset

