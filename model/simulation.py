import pandas as pd
from datetime import timedelta
import logging
from tqdm import tqdm
from model.user import User
import model.processing_utils as processing_utils
import numpy as np


class Simulation:
    def __init__(self, city_name, data_file, train_window, k_days, threshold, max_steps, recommender, topK,
                 p, data_version="small", category_attribute_name='Second_Category', seed=31,
                 selection_strategy='weighted'):

        self.seed = seed
        #np.random.seed(self.seed)
        self.city_name = city_name

        self.data_file = data_file
        self.data_version = data_version
        self.train_window = train_window  # Training window in days
        self.k_days = k_days  # Epoch length in days
        self.threshold = threshold
        self.current_time = None
        self.max_steps = max_steps
        self.step_counter = 0
        self.days_since_last_epoch = 0  # Tracks days since last epoch
        self.epoch_counter = 0  # Counts completed epochs
        self.min_activity = 5  # Minimum amount of POIs visited to be included in the training set
        self.recommender = recommender
        self.topK = topK
        self.selection_strategy = selection_strategy

        self.strange_return = 0
        self.strange_explore = 0
        self.strange_recommender = 0

        self.category_attribute_name = category_attribute_name

        # Parameters to automatically compute the probability of Returning against Exploring
        # Taken from https://www.nature.com/articles/ncomms9166
        self.ro = 0.6
        self.gamma = -0.21
        self.p = p

        # Load the data for the specified city
        self.load_city_data()

        self.users = {}
        self.new_interactions = []
        self.performance_history = []
        self.simulated_recommendation_lists = {}
        self.model_state = 'Simulation'

        self.prepare_datasets()

        # Get unique venue information, excluding time-related and visit-specific columns
        venue_columns = ['venueID', 'lat', 'lng', 'venue_descr', 'First_Category', 'Second_Category']
        self.unique_venue_data = self.train_data[venue_columns].drop_duplicates(subset=['venueID'])

        self.category_to_venues = self.unique_venue_data.groupby(self.category_attribute_name)['venueID'].agg(
            list).to_dict()  #correct category_attribute_name to venues
        self.first_category_to_venues = self.unique_venue_data.groupby('First_Category')['venueID'].agg(list).to_dict()
        self.second_category_to_venues = self.unique_venue_data.groupby('Second_Category')['venueID'].agg(
            list).to_dict()

        #save category_to_venues to a csv file
        category_to_venues_path = f"./cat2ven.csv"
        pd.DataFrame(self.category_to_venues.items(), columns=['category', 'venues']).to_csv(category_to_venues_path,
                                                                                             index=False)
        first_category_to_venues_path = f"./first_cat2ven.csv"
        pd.DataFrame(self.first_category_to_venues.items(), columns=['category', 'venues']).to_csv(
            first_category_to_venues_path, index=False)

        self.venue_info = self.unique_venue_data.set_index('venueID').to_dict('index')

        #save the venue_info to a csv file
        venue_info_path = f"./data/processed/venue_info_{self.city_name}_version_{self.data_version}_trainWindow_{self.train_window}.csv"
        self.unique_venue_data.to_csv(venue_info_path, index=False)

        self.create_users(selection_strategy)
        if self.selection_strategy == 'popularity-weighted':
            # Compute popularity of each poi
            self.venueid2popularity = (self.X_train != 0).sum(axis=0).to_dict()

        if self.p != 0:
            self.train_recommender_system()

    def load_city_data(self):
        # Load the data from the CSV file
        self.city_data = pd.read_csv(self.data_file)
        self.row_number = len(self.city_data)
        print(f'Dataset has {len(self.city_data)} records.')
        self.city_data['time'] = pd.to_datetime(self.city_data['time'])
        self.city_data['uid'] = self.city_data['uid'].astype(int)
        self.city_data['venueID'] = self.city_data['venueID'].astype(str)
        # Drop duplicates
        self.city_data = self.city_data.drop_duplicates()
        print(self.city_data.head())
        #self.city_name = os.path.basename(self.data_file).split('_')[0]
        print(f'Dataset has {len(self.city_data)} records after removing duplicates.')

    def prepare_datasets(self):
        # Set training period
        start_date = self.city_data['time'].min()
        training_duration = timedelta(days=self.train_window)
        self.t = start_date + training_duration
        self.current_time = self.t
        # Training data
        self.train_data = self.city_data[self.city_data['time'] < self.t]

        # Filtering out users with few interactions (visited less than X POIs)
        self.train_data = self.filtering_training_data(self.train_data)
        self.train_data.to_csv(
            f"./data/processed/training_data_city_{self.city_name}-{self.data_version}__trainWindow_{self.train_window}__.csv")

        self.distance_list = processing_utils.calculate_distance_list(self.train_data)

        self.radius_relevance = np.median(self.distance_list)

        #self.distance_matrix = processing_utils.calculate_distance_matrix(self.train_data, self.city_name, self.data_version, self.train_window)
        #print(f"Distance matrix calculated for {self.city_name}.")

        self.kd_tree, self.venue_ids = processing_utils.calculate_kd_tree(self.train_data)

        self.radius_df = processing_utils.calculate_radius_df(self.train_data, self.radius_relevance, self.city_name,
                                                              self.category_attribute_name, self.data_version,
                                                              self.train_window)
        print(f"Radius DataFrame calculated for {self.city_name}.")

        # Simulation data
        self.simulation_data = self.city_data[self.city_data['time'] >= self.t]
        # Excluding users not present in the training dataset
        self.simulation_data = self.simulation_data[self.simulation_data['uid'].isin(self.train_data['uid'].unique())]
        # Excluding venues not present in the training dataset
        self.simulation_data = self.simulation_data[
            self.simulation_data['venueID'].isin(self.train_data['venueID'].unique())]

        self.simulation_data.to_csv(
            f"./data/processed/simulation_data_city_{self.city_name}-{self.data_version}__trainWindow_{self.train_window}__.csv")

        # Prepare interaction matrix
        self.X_train = self.create_interaction_matrix(self.train_data)

    def filtering_training_data(self, training_dataset):
        # Count the number of rows per user
        user_counts = training_dataset['uid'].value_counts()
        # Get the users with at least x rows
        valid_users = user_counts[user_counts >= self.min_activity].index
        # Filter the DataFrame to include only valid users
        filtered_dataset = training_dataset[training_dataset['uid'].isin(valid_users)]
        return filtered_dataset

    def create_interaction_matrix(self, data):
        interaction_counts = data.groupby(['uid', 'venueID']).size().reset_index(name='counts')
        interaction_matrix = interaction_counts.pivot(index='uid', columns='venueID', values='counts').fillna(0)
        return interaction_matrix

    def create_users(self, selection_strategy):
        user_ids = self.train_data['uid'].unique()
        for uid in user_ids:
            user_history = self.train_data[self.train_data['uid'] == uid]
            user = User(uid, user_history, self.p, self, selection_strategy=selection_strategy)
            self.users[uid] = user

    def train_recommender_system(self):
        self.recommender.fit(self.X_train)
        print(f"Recsys trained for {self.city_name}.")

    def run(self):

        with tqdm(total=self.max_steps, desc='Simulation Progress', ncols=80, leave=True) as pbar:

            while self.model_state == 'Simulation' and self.step_counter < self.max_steps:
                self.step()
                pbar.update(1)
                if self.days_since_last_epoch % self.k_days == 0:
                    self.epoch_counter += 1  # Increment epoch counter
                    print(f"strange return: {self.strange_return}")
                    print(f"strange explore: {self.strange_explore}")
                    print(f"strange recommender: {self.strange_recommender}")

    def step(self):
        # active_users is a dictionary where:
        # - Key: user ID (uid)
        # - Value: dictionary containing three levels of category hierarchies:
        #   {
        #     'First_Level': ['Food', 'Building', ...],           # Broad categories
        #     'Second_Level': ['Restaurant', 'Gym', ...], # More specific categories
        #   'venueID': ['1234', '5678', ...] # List of visited venues
        #   }

        active_users = self.activate_users()
        print(f"step {self.step_counter}: {len(active_users)} active users, epoch {self.epoch_counter}")
        logging.info(f"step {self.step_counter}: {len(active_users)} active users")

        # Process interactions for each active user
        # For each user, we pass their entire category hierarchy to process_interactions()
        # This allows the user object to analyze interactions at different category granularity levels
        for uid, category_hierarchy in active_users.items():
            user = self.users[uid]
            user.process_interactions(category_hierarchy)

        # Advance time by one day
        self.current_time += timedelta(days=1)
        self.step_counter += 1
        self.days_since_last_epoch += 1  # Increment days since last epoch

        # Check if an epoch has been completed
        if self.days_since_last_epoch >= self.k_days:
            self.validate_performance()
            self.days_since_last_epoch = 0  # Reset days since last epoch

        # Check if maximum steps have been reached
        if self.step_counter >= self.max_steps:
            self.model_state = 'Stop'
            print(f"Maximum simulation steps reached for {self.city_name}. Simulation will stop.")

        # Check if simulation should stop due to data availability
        elif self.current_time > self.simulation_data['time'].max():
            self.model_state = 'Stop'
            print(f"No more simulation data available for {self.city_name}. Simulation will stop.")

    def activate_users(self):
        # Get data for current day
        daily_data = self.simulation_data[self.simulation_data['time'].dt.date == self.current_time.date()]

        # Create a dictionary to store hierarchical categories for each user
        user_categories = {}

        # Group by user ID
        for uid in daily_data['uid'].unique():
            user_data = daily_data[daily_data['uid'] == uid]

            user_categories[uid] = {
                "First_Category": user_data['First_Category'].tolist(),
                "Second_Category": user_data['Second_Category'].tolist(),
                "venueID": user_data['venueID'].tolist()
            }

        return user_categories

    def check_simulation_stopping_criterion(self):
        pass

    def validate_performance(self):
        # current_performance = self.evaluate_model_performance()
        current_performance = -1.0
        self.performance_history.append(current_performance)
        # Decide whether to retrain or stop based on performance
        self.check_simulation_stopping_criterion()
        # Update training data and retrain model
        if self.p != 0:
            self.update_training_data()
            self.train_recommender_system()

    def evaluate_model_performance(self):
        test_data = self.simulation_data[self.simulation_data['time'].dt.date > self.current_time.date()]
        test_data = test_data[test_data['time'].dt.date <= self.current_time.date() + timedelta(self.k_days)]
        num_hits, num_recommendations = 0, 0
        for index, row in test_data.iterrows():
            user = self.users[row['uid']]
            if row['venueID'] in set(user.history['venueID']):
                continue
            venue_category = row[self.category_attribute_name]
            recommendation_list = self.recommender.return_topK(user, self.topK, venue_category, self.category_to_venues)

            num_hits += row['venueID'] in recommendation_list
            num_recommendations += 1

        num_interactions = num_recommendations
        if num_interactions == 0:
            print(f"No interactions for {self.city_name}.")
            return 999
        hit_rate = num_hits / num_recommendations
        print(
            f"Epoch {self.epoch_counter}: {num_interactions} interactions; {round(hit_rate, 2)} hit rate; {num_recommendations} recommendations.")
        return hit_rate

    def update_training_data(self):
        if len(self.new_interactions) > 0:
            # Convert list of dictionaries to DataFrame
            new_data = pd.DataFrame(self.new_interactions)
            # Concatenate new data to the training data
            self.train_data = pd.concat([self.train_data, new_data], ignore_index=True)
            # Update the interaction matrix
            self.X_train = self.create_interaction_matrix(self.train_data)
            # Clear the list
            self.new_interactions = []
            print(f"Training data updated for {self.city_name}.")
        else:
            print(f"No new interactions to update for {self.city_name}.")
