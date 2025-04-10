import pandas as pd
import numpy as np
import sys
from scipy import stats
from scipy.stats import entropy
from collections import Counter

from tqdm import tqdm, tqdm_notebook

tqdm_notebook().pandas()

from math import pi, sin, cos, sqrt, atan2

#earthradius = 6371.0
earthradius = 6371000

def getDistanceByHaversine(loc1, loc2):
    "Haversine formula - give coordinates as (lat_decimal,lon_decimal) tuples"

    lat1, lon1 = loc1
    lat2, lon2 = loc2

    # convert to radians
    lon1 = lon1 * pi / 180.0
    lon2 = lon2 * pi / 180.0
    lat1 = lat1 * pi / 180.0
    lat2 = lat2 * pi / 180.0

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2.0)) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
    km = earthradius * c
    return km


def _random_entropy_individual(traj):
    """
    Compute the random entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individual

    Returns
    -------
    float
        the random entropy of the individual 
    """
    n_distinct_locs = len(traj.groupby(["lat", "lon"]))
    entropy = np.log2(n_distinct_locs)
    return entropy


def random_entropy(traj, show_progress=False):
    """Random entropy.

    Compute the random entropy of a set of individuals in a TrajDataFrame.
    The random entropy of an individual :math:`u` is defined as [EP2009]_ [SQBB2010]_: 

    .. math::
        E_{rand}(u) = log_2(N_u)

    where :math:`N_u` is the number of distinct locations visited by :math:`u`, capturing the degree of predictability of :math:`u`’s whereabouts if each location is visited with equal probability. 

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.

    Returns
    -------
    pandas DataFrame
        the random entropy of the individuals.
    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_random_entropy_individual(traj)], columns=[sys._getframe().f_code.co_name])

    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _random_entropy_individual(x))
    else:
        df = traj.groupby("uid").apply(lambda x: _random_entropy_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _uncorrelated_entropy_individual(traj, normalize=False):
    """
    Compute the uncorrelated entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    normalize : boolean, optional
        if True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations visited by individual :math:`u`. The default is False.

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    n = len(traj)
    probs = [1.0 * len(group) / n for group in traj.groupby(by=["lat", "lon"]).groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(traj[["lat", "lon"]].values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def uncorrelated_entropy(traj, normalize=False, show_progress=False):
    """Uncorrelated entropy.

    Compute the temporal-uncorrelated entropy of a set of individuals in a TrajDataFrame. The temporal-uncorrelated entropy of an individual :math:`u` is defined as [EP2009]_ [SQBB2010]_ [PVGSPG2016]_: 

    .. math::
        E_{unc}(u) = - \sum_{j=1}^{N_u} p_u(j) log_2 p_u(j)

    where :math:`N_u` is the number of distinct locations visited by :math:`u` and :math:`p_u(j)` is the historical probability that a location :math:`j` was visited by :math:`u`. The temporal-uncorrelated entropy characterizes the heterogeneity of :math:`u`'s visitation patterns.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    normalize : boolean, optional
        if True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations visited by individual :math:`u`. The default is False.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.

    Returns
    -------
    pandas DataFrame
        the temporal-uncorrelated entropy of the individuals.

    """
    column_name = sys._getframe().f_code.co_name
    if normalize:
        column_name = 'norm_%s' % sys._getframe().f_code.co_name

    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_uncorrelated_entropy_individual(traj)], columns=[column_name])

    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _uncorrelated_entropy_individual(x, normalize=normalize))
    else:
        df = traj.groupby("uid").apply(lambda x: _uncorrelated_entropy_individual(x, normalize=normalize))
    return pd.DataFrame(df).reset_index().rename(columns={0: column_name})


def _radius_of_gyration_individual(traj):
    """
    Compute the radius of gyration of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectory of the individual.

    Returns
    -------
    float
        the radius of gyration of the individual.
    """
    lats_lngs = traj[["lat", "lon"]].values
    center_of_mass = np.mean(lats_lngs, axis=0)
    rg = np.sqrt(np.mean([getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0 for lat, lng in lats_lngs]))
    return rg


def radius_of_gyration(traj, show_progress=False):
    """Radius of gyration.

    Compute the radii of gyration (in meters) of a set of individuals in a TrajDataFrame.
    The radius of gyration of an individual :math:`u` is defined as [GHB2008]_ [PRQPG2013]_: 

    .. math:: 
        r_g(u) = \sqrt{ \\frac{1}{n_u} \sum_{i=1}^{n_u} dist(r_i(u) - r_{cm}(u))^2}

    where :math:`r_i(u)` represents the :math:`n_u` positions recorded for :math:`u`, and :math:`r_{cm}(u)` is the center of mass of :math:`u`'s trajectory. In mobility analysis, the radius of gyration indicates the characteristic distance travelled by :math:`u`.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.

    Returns
    -------
    pandas DataFrame
        the radius of gyration of each individual.

    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_radius_of_gyration_individual(traj)], columns=[sys._getframe().f_code.co_name])

    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _radius_of_gyration_individual(x))
    else:
        df = traj.groupby("uid").apply(lambda x: _radius_of_gyration_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name}) 




def _k_radius_of_gyration_individual(traj, k=2):
    """Compute the k-radius of gyration of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individual.
    
    k : int, optional
        the number of most frequent locations to consider. The default is 2. The possible range of values is math:`[2, +inf]`.
    
    Returns
    -------
    float
        the k-radius of gyration of the individual. 
    """
    traj['visits'] = traj.groupby(["lat", "lon"]).transform('count')["time"]
    top_k_locations = traj.drop_duplicates(subset=["lat", "lon"]).sort_values(by=['visits', "time"],
                                                                              ascending=[False, True])[:k]
    visits = top_k_locations['visits'].values
    total_visits = sum(visits)
    lats_lngs = top_k_locations[["lat", "lon"]].values

    center_of_mass = visits.dot(lats_lngs) / total_visits
    krg = np.sqrt(sum([visits[i] * (getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0)
                       for i, (lat, lng) in enumerate(lats_lngs)]) / total_visits)
    return krg


def k_radius_of_gyration(traj, k=2, show_progress=True):
    """k-radius of gyration.
    
    Compute the k-radii of gyration (in kilometers) of a set of individuals in a TrajDataFrame.
    The k-radius of gyration of an individual :math:`u` is defined as [PSRPGB2015]_:
    
    .. math::
        r_g^{(k)}(u) = \sqrt{\\frac{1}{n_u^{(k)}} \sum_{i=1}^k (r_i(u) - r_{cm}^{(k)}(u))^2} 
        
    where :math:`r_i(u)` represents the :math:`n_u^{(k)}` positions recorded for :math:`u` on their k most frequent locations, and :math:`r_{cm}^{(k)}(u)` is the center of mass of :math:`u`'s trajectory considering the visits to the k most frequent locations only. In mobility analysis, the k-radius of gyration indicates the characteristic distance travelled by that individual as induced by their k most frequent locations.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individual.
    
    k : int, optional
        the number of most frequent locations to consider. The default is 2. The possible range of values is :math:`[2, +inf]`.
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the k-radii of gyration of the individuals

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.individual import k_radius_of_gyration
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> krg_df = k_radius_of_gyration(tdf)
    >>> print(krg_df.head())
       uid  3k_radius_of_gyration
    0    0               7.730516
    1    1               3.620671
    2    2               6.366549
    3    3              10.543072
    4    4            3910.808802

    References
    ----------
    .. [PSRPGB2015] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F. & Barabasi, A. L. (2015) Returners and Explorers dichotomy in human mobility. Nature Communications 6, https://www.nature.com/articles/ncomms9166
    
    See Also
    --------
    radius_of_gyration
    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_k_radius_of_gyration_individual(traj, k=k)], columns=['%s%s' % (k, sys._getframe().f_code.co_name)])
    
    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _k_radius_of_gyration_individual(x, k))
    else:
        df = traj.groupby("uid").apply(lambda x: _k_radius_of_gyration_individual(x, k))
    return pd.DataFrame(df).reset_index().rename(columns={0: '%s%s' % (k, sys._getframe().f_code.co_name)})














def _true_entropy(sequence):
    n = len(sequence)

    # these are the first and last elements
    sum_lambda = 1. + 2.

    def in_seq(a, b):
        for i in range(len(a) - len(b) + 1):
            valid = True
            for j, v in enumerate(b):
                if a[i + j] != v:
                    valid = False
                    break
            if valid: return True
        return False

    for i in range(1, n - 1):
        j = i + 1
        while j < n and in_seq(sequence[:i], sequence[i:j]):
            j += 1
        if j == n: j += 1     # EOF character
        sum_lambda += j - i

    return 1. / sum_lambda * n * np.log2(n)


def _real_entropy_individual(traj):
    """
    Compute the real entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individual.
    
    Returns
    -------
    float
        the real entropy of the individual.
    """
    time_series = tuple(map(tuple, traj[["lat", "lon"]].values))
    entropy = _true_entropy(time_series)
    return entropy


def real_entropy(traj, show_progress=False):
    """Real entropy.
    
    Compute the real entropy of a set of individuals in a TrajDataFrame. 
    The real entropy of an individual :math:`u` is defined as [SQBB2010]_: 
    
    .. math:: 
        E(u) = - \sum_{T'_u}P(T'_u)log_2[P(T_u^i)]
    
    where :math:`P(T'_u)` is the probability of finding a particular time-ordered subsequence :math:`T'_u` in the trajectory :math:`T_u`. The real entropy hence depends not only on the frequency of visitation, but also the order in which the nodes were visited and the time spent at each location, thus capturing the full spatiotemporal order present in an :math:`u`'s mobility patterns.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals
    
    show_progress : boolean, optional 
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the real entropy of the individuals
    
    Warning
    -------
    The input TrajDataFrame must be sorted in ascending order by `datetime`. Note that the computation of this measure is, by construction, slow.
    
    """

    #if two consecutive rows have the same time, we add a small value to the time of the second row
    traj["time"] = pd.to_datetime(traj["time"])
    # Create a mask for duplicate timestamps
    mask = traj["time"].duplicated()

    # Only modify the duplicated timestamps
    if mask.any():
        # Get indices of duplicated times
        dup_indices = mask[mask].index
        
        # Add a small increment (e.g., 1 millisecond) to each duplicate
        for idx in dup_indices:
            traj.loc[idx, "time"] += pd.Timedelta(milliseconds=1)


    #check if the TrajDataFrame is sorted by "time" column
    #if not traj["time"].is_monotonic_increasing:
    #    print(traj["time"])
    #    raise ValueError("The TrajDataFrame must be sorted in ascending order by 'datetime' column.")

    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_real_entropy_individual(traj)], columns=[sys._getframe().f_code.co_name])
    





    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _real_entropy_individual(x))
    else:
        df = traj.groupby("uid").apply(lambda x: _real_entropy_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})










def place_sequence_entropy(traj, show_progress=False):
    """
    Compute an entropy measure that focuses only on:
    1. The uniqueness of places visited (spatial diversity)
    2. The simple sequence of visits (which place follows which)
    
    Parameters
    ----------
    traj : TrajDataFrame
        The trajectories of individuals with columns 'lat', 'lon', 'time', and optionally 'uid'
    show_progress : boolean, optional
        If True, show a progress bar. Default is True.
        
    Returns
    -------
    pandas DataFrame
        The place and sequence entropy of individuals
    """
    # Ensure time column is datetime
    traj["time"] = pd.to_datetime(traj["time"])
    
    # Handle duplicate timestamps
    mask = traj["time"].duplicated()
    if mask.any():
        dup_indices = mask[mask].index
        for idx in dup_indices:
            traj.loc[idx, "time"] += pd.Timedelta(milliseconds=1)
    
    # Sort by time to ensure correct sequence
    traj = traj.sort_values(by="time")
    
    def _compute_individual_entropy(user_traj):
        # Create location identifier (combining lat/lon)
        user_traj['location_id'] = user_traj.apply(
            lambda row: f"{row['lat']:.6f}_{row['lon']:.6f}", axis=1
        )
        
        # 1. Place entropy - how diverse are the places visited?
        location_counts = Counter(user_traj['location_id'])
        location_probs = np.array(list(location_counts.values())) / len(user_traj)
        place_ent = entropy(location_probs, base=2)
        
        # 2. Sequence entropy - captures which place follows which
        if len(user_traj) > 1:
            transitions = [f"{a}_to_{b}" for a, b in zip(
                user_traj['location_id'].iloc[:-1], 
                user_traj['location_id'].iloc[1:]
            )]
            
            trans_counts = Counter(transitions)
            trans_probs = np.array(list(trans_counts.values())) / len(transitions)
            sequence_ent = entropy(trans_probs, base=2)
            
            # Combine place and sequence entropy with equal weights
            combined_entropy = 0.5 * place_ent + 0.5 * sequence_ent
        else:
            # If only one location, sequence entropy doesn't apply
            combined_entropy = place_ent
        
        return {
            'place_entropy': place_ent,
            'sequence_entropy': sequence_ent if len(user_traj) > 1 else 0,
            'combined_entropy': combined_entropy
        }
    
    # If 'uid' column is not present in the TrajDataFrame
    if "uid" not in traj.columns:
        result = _compute_individual_entropy(traj)
        return pd.DataFrame([result])
    
    # Calculate for each individual
    if show_progress:
        try:
            from tqdm import tqdm
            tqdm.pandas(desc="Computing entropy")
            df = traj.groupby("uid").progress_apply(_compute_individual_entropy)
        except ImportError:
            df = traj.groupby("uid").apply(_compute_individual_entropy)
    else:
        df = traj.groupby("uid").apply(_compute_individual_entropy)
    
    # Convert list of dictionaries to DataFrame
    return pd.DataFrame(df.tolist(), index=df.index).reset_index()






def _number_of_locations_individual(traj):
    """
    Compute the number of visited locations of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individual.
    
    Returns
    -------
    int
        number of distinct locations visited by the individual.
    """
    n_locs = len(traj.groupby(["lat", "lon"]).groups)
    return n_locs


def number_of_locations(traj, show_progress=False):
    """Number of distinct locations.
    
    Compute the number of distinct locations visited by a set of individuals in a TrajDataFrame [GHB2008]_.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the number of distinct locations visited by the individuals.
    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_number_of_locations_individual(traj)], columns=[sys._getframe().f_code.co_name])
    
    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _number_of_locations_individual(x))
    else:
        df = traj.groupby("uid").apply(lambda x: _number_of_locations_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})



def number_of_visits(traj, show_progress=False):
    """Number of visits.
    
    Compute the number of visits (i.e., data points) for each individual in a TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the number of visits or points per each individual.
    
    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return len(traj)
    
    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: len(x))
    else:
        df = traj.groupby("uid").apply(lambda x: len(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _location_frequency_individual(traj, normalize=True,
                                   location_columns=["lat", "lon"]):
    """
    Compute the visitation frequency of each location for a single individual given their TrajDataFrame.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectory of the individual.
    
    normalize : boolean, optional
        if True, compute the ratio of visits, otherwise the row count of visits to each location. The default is True.
    
    location_columns : list, optional
        the name of the column(s) indicating the location. The default is ["lat", "lon"].
    
    Returns
    -------
    pandas DataFrame
        the location frequency of each location of the individual. 
    """
    freqs = traj.groupby(location_columns).count()["time"].sort_values(ascending=False)
    if normalize:
        freqs /= freqs.sum()
    return freqs


def location_frequency(traj, normalize=True, as_ranks=False, show_progress=False,
                       location_columns=["lat", "lon"]):
    """Location frequency.
    
    Compute the visitation frequency of each location, for a set of individuals in a TrajDataFrame. Given an individual :math:`u`, the visitation frequency of a location :math:`r_i` is the number of visits to that location by :math:`u`. The visitation frequency :math:`f(r_i)` of location :math:`r_i` is also defined in the literaure as the probability of visiting location :math:`r_i` by :math:`u` [SKWB2010]_ [PF2018]_:
    
    .. math::
        f(r_i) = \\frac{n(r_i)}{n_u}
        
    where :math:`n(r_i)` is the number of visits to location :math:`r_i` by :math:`u`, and :math:`n_u` is the total number of data points in :math:`u`'s trajectory.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    normalize : boolean, optional
        if True, the number of visits to a location by an individual is computed as probability, i.e., divided by the individual's total number of visits. The default is True.
    
    as_ranks : boolean, optional
        if True, return a list where element :math:`i` indicates the average visitation frequency of the :math:`i`-th most frequent location. The default is False.
   
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
        
    location_columns : list, optional
        the name of the column(s) indicating the location. The default is ["lat", "lon"].
    
    Returns
    -------
    pandas DataFrame or list
        the location frequency for each location for each individual, or the ranks list for each individual.
    
    """
    # TrajDataFrame without 'uid' column
    if "uid" not in traj.columns: 
        df = pd.DataFrame(_location_frequency_individual(traj, location_columns=location_columns))
        return df.reset_index()
    
    # TrajDataFrame with a single user
    n_users = len(traj["uid"].unique())
    if n_users == 1: # if there is only one user in the TrajDataFrame
        df = pd.DataFrame(_location_frequency_individual(traj, location_columns=location_columns))
        return df.reset_index()
    
    # TrajDataFrame with multiple users
    if show_progress:
        df = pd.DataFrame(traj.groupby("uid")
                          .progress_apply(lambda x: _location_frequency_individual(x, normalize=normalize, location_columns=location_columns)))
    else:
        df = pd.DataFrame(traj.groupby("uid")
                          .apply(lambda x: _location_frequency_individual(x, normalize=normalize, location_columns=location_columns)))
    
    df = df.rename(columns={"time": 'location_frequency'})
    
    if as_ranks:
        ranks = [[] for i in range(df.groupby('uid').count().max().location_frequency)]
        for i, group in df.groupby('uid'):
            for j, (index, row) in enumerate(group.iterrows()):
                ranks[j].append(row.location_frequency)
        ranks = [np.mean(rr) for rr in ranks]
        return ranks
    
    return df




def location_diversity(traj):
    """
    Calculate the ratio of unique locations to total visits
    """
    unique_locations = traj[['lat', 'lon']].drop_duplicates().shape[0]
    total_visits = traj.shape[0]
    return unique_locations / total_visits




######################
######################
######################
######################
######COLLECTIVE######


def collective_location_entropy(merged_histories):
    """
    Calculate entropy based on the distribution of visits across all locations
    Higher values mean more evenly distributed visits across locations
    """
    location_counts = merged_histories.groupby(['lat', 'lon']).size()
    total_visits = len(merged_histories)
    probabilities = location_counts / total_visits
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def user_distribution_entropy(merged_histories):
    """
    Measures how users are distributed across locations
    Higher values mean users are spread more evenly across locations
    """
    # Count unique users per location
    users_per_location = merged_histories.groupby(['lat', 'lon'])['uid'].nunique()
    total_user_visits = users_per_location.sum()
    probabilities = users_per_location / total_user_visits
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy