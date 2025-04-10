import numpy as np
import pandas as pd
import model.processing_utils as processing_utils
import logging
import os


class User:
    def __init__(self, uid, history, p, simulation, selection_strategy='random'):
        self.uid = uid
        self.history = history  # User's historical interactions
        self.p = p  # Probability of accepting a recommendation
        self.simulation = simulation
        self.state = None

        self.random_state = np.random.RandomState(seed=self.simulation.seed + uid)
        self.selection_strategy = selection_strategy

        logging.basicConfig(
        filename=f"logs/Seed_{self.simulation.seed}_p_{self.simulation.p}.log",
        filemode='a',  # Append mode to keep log history
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',  # Ensures timestamp for each line
    )
    

    def process_interactions(self, category_hierarchy):
        # category_hierarchy is a dictionary with three levels of categories
        #   {
        #     'First_Category': ['Food', 'Building', ...],           # Broad categories
        #     'Second_Category': ['Restaurant', 'Gym', ...], # More specific categories
        #     'venueID': ['1234', '5678', ...] # Venue IDs
        #   }

        # Get the number of interactions (all levels should have same length)
        num_interactions = len(category_hierarchy['First_Category'])

        # Process each interaction at all category levels
        for i in range(num_interactions):
            interaction_categories = {
                'First_Category': category_hierarchy['First_Category'][i],
                'Second_Category': category_hierarchy['Second_Category'][i],
                'venueID': category_hierarchy['venueID'][i]
            }
            self.make_decision(interaction_categories)

    def make_decision(self, category_hierarchy, simulated=False):
        """
        Make a decision about which venue to visit based on the hierarchical category structure.
        """
        # Get the number of distinct venues visited by the user
        num_distinct_venues = len(np.unique(self.history['venueID'].values))
        # Get the correct category based on the category_attribute_name from simulation
        correct_category = category_hierarchy[self.simulation.category_attribute_name]
        current_lat = self.simulation.train_data.loc[
            self.simulation.train_data['venueID'] == self.history['venueID'].values[-1], 'lat'].iloc[0]
        current_lng = self.simulation.train_data.loc[
            self.simulation.train_data['venueID'] == self.history['venueID'].values[-1], 'lng'].iloc[0]


        rand_decision = self.random_state.rand()
        venues_correct_category_at_radius = []
        radius = -1.0
        while len(venues_correct_category_at_radius) == 0: #at least a jump that holds a venue
            #pick a weighted random radius from self.simulation.distance_list
            radius = processing_utils.pick_radius_weighted(self.simulation.distance_list, self.random_state)
            # Get nearby venues within radius
            venues_at_radius = processing_utils.get_nearby_venues(
                current_lat, current_lng,
                radius,
                self.simulation.train_data
            )
            venues_correct_category_at_radius = [venue for venue in venues_at_radius
                                if venue in self.simulation.category_to_venues[category_hierarchy[self.simulation.category_attribute_name]] ]
            

        if self.random_state.rand() < self.p:
            self.state = 'Rec'
            venue_id = self.get_recommended_venue(correct_category, category_hierarchy=category_hierarchy, 
                                                  radius=radius, venues_correct_category_at_radius=venues_correct_category_at_radius)
        else:
            # Decide between 'Explore' and 'Return'
            if self.random_state.rand() >= self.simulation.ro * num_distinct_venues ** self.simulation.gamma:
                self.state = 'Return'
                venue_id = self.get_return_venue(category_hierarchy,
                                                #just for strange returners
                                                radius=radius, venues_correct_category_at_radius=venues_correct_category_at_radius,
                                                venues_at_radius=venues_at_radius)
            else:
                self.state = 'Explore'
                # Pass the complete category hierarchy to get_new_venue
                venue_id = self.get_new_venue(category_hierarchy, 
                                              radius=radius, venues_correct_category_at_radius=venues_correct_category_at_radius,
                                              venues_at_radius=venues_at_radius)

        if not simulated:
            if venue_id is None:
                print(self.state)
                print('')
            # Record interaction
            self.record_interaction(venue_id)
        else:
            return venue_id

    def record_interaction(self, venue_id):
        venue_info = self.simulation.venue_info[venue_id]

        new_interaction = {
            'uid': self.uid,
            'venueID': venue_id,
            'time': self.simulation.current_time,  # Added at interaction time
            'step': self.simulation.step_counter,  # Added at interaction time
            'epoch': self.simulation.epoch_counter,  # Added at interaction time
            # Static venue info
            'lat': venue_info['lat'],
            'lng': venue_info['lng'],
            'venue_descr': venue_info['venue_descr'],
            'First_Category': venue_info['First_Category'],
            'Second_Category': venue_info['Second_Category'],
            'city': self.simulation.city_name,
            'simulation_state': self.state,
        }

        logging.info(f"User {self.uid} - Step {self.simulation.step_counter} - Venue {venue_id} - State {self.state}")

        # Append the new interaction to the list
        self.simulation.new_interactions.append(new_interaction)
        # Convert the new interaction to a DataFrame
        new_interaction_df = pd.DataFrame([new_interaction])
        # Concatenate the new interaction to the user's history DataFrame
        self.history = pd.concat([self.history, new_interaction_df], ignore_index=True)

    def select_item_from_recommendation_list(self, recommendation_list):
        if self.selection_strategy == 'random':
            return self.random_state.choice(list(recommendation_list.keys()))
        elif self.selection_strategy == 'popular':
            return list(recommendation_list.keys())[0]
        elif self.selection_strategy == 'popularity-weighted':
            recommendation_list = {key: self.simulation.venueid2popularity[key] for key in recommendation_list}
            return processing_utils.sample_weighted(recommendation_list, self.random_state)
        elif self.selection_strategy == 'position-bias':
            raise NotImplementedError()
        elif self.selection_strategy == 'weighted':
            # Get min and max values
            min_value = min(recommendation_list.values())
            max_value = max(recommendation_list.values())

            # Avoid division by zero if all values are the same
            if max_value == min_value:
                # If all values are the same, assign equal probability (e.g., 1/N)
                recommendation_list = {key: 1 / len(recommendation_list) for key in recommendation_list}
            else:
                # Apply Min-Max Scaling to transform values into the [0,1] range
                recommendation_list = {
                    key: (value - min_value) / (max_value - min_value)
                    for key, value in recommendation_list.items()
                }

            return processing_utils.sample_weighted(recommendation_list, self.random_state)



    def get_recommended_venue(self, venue_category, category_hierarchy=None, radius=None, venues_correct_category_at_radius=None):

        if self.uid in self.simulation.X_train.index:
            
            recommendation_list = self.simulation.recommender.return_topK(self, venues_correct_category_at_radius, self.simulation.topK, venue_category,
                                                                          self.simulation.category_to_venues)

            if len(recommendation_list) == 0: #shouldn't happen
                    #logging.info(f'Step: {self.simulation.step_counter}. Recommender. No venues in the category. Returning.') 
                    #user visited all venues als in the first category
                    self.simulation.strange_recommender += 1
                    chosen_return_venue = self.get_return_venue(category_hierarchy=category_hierarchy, radius=radius, venues_correct_category_at_radius=venues_correct_category_at_radius)
                    self.state = "Return"
                    return chosen_return_venue
                

            if self.uid in self.simulation.simulated_recommendation_lists:
                self.simulation.simulated_recommendation_lists[self.uid].append(recommendation_list)
            else:
                self.simulation.simulated_recommendation_lists[self.uid] = [recommendation_list]

            chosen_venue = self.select_item_from_recommendation_list(recommendation_list)
            return chosen_venue
        else:
            raise Exception(f'User {self.uid} is not present in the dataset')




    def get_return_venue(self, category_hierarchy = None, radius = None, venues_correct_category_at_radius=None, venues_at_radius=None):

        def _get_return_venue(list_of_venue_ids):       
                # Count the occurrences of each VenueID
                venue_counts = self.history[self.history['venueID'].isin(list_of_venue_ids)]['venueID'].value_counts()

                # Get the VenueIDs and their respective probabilities
                venue_ids = venue_counts.index
                weights = venue_counts.values
                # Select the subset of venueIDs of that particular category
                weights = np.array([weight for idx, weight in enumerate(weights)])
                venue_ids = np.array([venue_id for venue_id in venue_ids])

                # Return a random choice weighted by visit frequencies
                return self.random_state.choice(venue_ids, p=weights / weights.sum())
        
        # Return to a random previously visited location with probability proportional to visit frequency
        if not self.history.empty:

            # Get venues for each category level using the simulation's category mappings
            venues_second_category = [venue for venue in self.history['venueID'].values
                                    if venue in self.simulation.second_category_to_venues[
                                        category_hierarchy['Second_Category']]]
            venues_first_category = [venue for venue in self.history['venueID'].values
                                    if venue in self.simulation.first_category_to_venues[
                                        category_hierarchy['First_Category']]]

            # Try to find venues starting from the target category level
                
            if self.simulation.category_attribute_name == "Second_Category":
                if len(venues_second_category) > 0:
                    venue_to_return = _get_return_venue(venues_second_category)
                    return venue_to_return
                elif len(venues_first_category) > 0:
                    venue_to_return = _get_return_venue(venues_first_category)
                    return venue_to_return
                
            elif self.simulation.category_attribute_name == "First_Category":
                if len(venues_first_category) > 0:
                    venue_to_return = _get_return_venue(venues_first_category)
                    return venue_to_return

            #if you arrived here, means that in your history no venue in the category hierarchy was previosly visited by you
            # it means that you must explore  
            self.simulation.strange_return+=1  
            venue_to_return = self.get_new_venue(category_hierarchy, radius, venues_correct_category_at_radius, venues_at_radius ) 
            self.state = "Explore"
            return venue_to_return

            
        else:
            raise Exception('Problem with history.')






    def get_new_venue(self, category_hierarchy, radius=None, venues_correct_category_at_radius=None, venues_at_radius=None):
        """
        Find a new venue to visit based on hierarchical category structure.
        """

        # drop the venues that are already visited
        venues_correct_category_at_radius = [venue for venue in venues_correct_category_at_radius if venue not in self.history["venueID"].values]
        current_position_venueID = self.history['venueID'].values[-1]

        if len(venues_correct_category_at_radius) == 0:
            #check if there are venues in the larger category in the radius
            if self.simulation.category_attribute_name == "Second_Category":
                venues_first_category_at_radius = [venue for venue in venues_at_radius
                                if venue in self.simulation.first_category_to_venues[category_hierarchy['First_Category']] ]
                venues_first_category_at_radius = [venue for venue in venues_first_category_at_radius if venue not in self.history["venueID"].values]
                if len(venues_first_category_at_radius) > 0:
                    #logging.info(f'Step: {self.simulation.step_counter} Explore. Correct Category: {category_hierarchy[self.simulation.category_attribute_name]}. Radius: {radius}. No venues in the correct category. Fallback to larger category.')
                    return processing_utils.select_weighted_venues_from_radius(
                        venues_first_category_at_radius, self.simulation.radius_df, column="total_nearby"
                    )


        else:
            return processing_utils.select_weighted_venues_from_radius(
                venues_correct_category_at_radius, self.simulation.radius_df, column="total_nearby"
            )


        # If no venue found in any category level, pick the nearest venue at correct category level. 
        self.simulation.strange_explore += 1
        correct_category = category_hierarchy[self.simulation.category_attribute_name]
        venuesOfCategory = self.simulation.category_to_venues[correct_category]

        history_set = set()
        nearest_venue = processing_utils.find_nearest_venue(
            kd_tree=self.simulation.kd_tree,
            venue_ids=self.simulation.venue_ids,
            venues_of_category=set(venuesOfCategory),
            current_venue_id=current_position_venueID,
            user_history=history_set  # Pass history to exclude visited venues
        )

        # if nearest_venue:
        #     logging.info(f'Step: {self.simulation.step_counter}. Explore. No venues in the correct category. Fallback to nearest venue.')
        # else:
        #     logging.info(f'Step: {self.simulation.step_counter}. Explore. No venues in the correct category. Fallback to nearest venue FAILED.')


        return nearest_venue
