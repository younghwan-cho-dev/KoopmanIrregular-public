import numpy as np
import pysindy as ps
import yaml
import pickle
import os

from datetime import datetime

"""
Utility functions for configuration management, preprocessing, evaluation, and file I/O.

Functions
---------
load_config(config_file)
    Load a YAML configuration file into a Python dictionary.

create_experiment_dir(base_dir="results")
    Create a timestamped directory for saving experiment outputs.

preprocess_data(train_data, observation_time_train_data, params)
    Convert time-series state trajectories into observable space and compute temporal differences.
    
    The function uses `pysindy`, a Python library for sparse identification of nonlinear dynamics.
    
    Brian M. de Silva, Kathleen Champion, Markus Quade, Jean-Christophe Loiseau, J. Nathan Kutz, and Steven L. Brunton., (2020). 
    PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. 
    Journal of Open Source Software, 5(49), 2104, https://doi.org/10.21105/joss.02104

    Kaptanoglu et al., (2022). 
    PySINDy: A comprehensive Python package for robust sparse system identification. 
    Journal of Open Source Software, 7(69), 3994, https://doi.org/10.21105/joss.03994

evaluate(analysis, feature_names)
    Extract final predicted eigenvalues and coefficients from optimization results.

save_pickle(data, path)
    Save a Python object to disk using pickle serialization.

Notes
-----
- Preprocessing is designed to support SINDy-based dictionary construction using polynomial and/or Fourier libraries.

Author: Younghwan Cho
"""


def load_config(config_file):
    """Load the configuration file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def create_experiment_dir(base_dir="results"):
    """Create a directory for the experiment based on the current timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def preprocess_data(
    train_data, observation_time_train_data, params
):
    """
    Preprocess the dataset.
    States are transformed into observables and time differences between consequent observations are computed.
    Then, for each trajectory, the observables and time differeneces are stacked along the row axis.

    Args:
        train_data (list): a list of 2-dimensional numpy array of the shape (observations, states).
        observation_time_train_data (list): a list of a numpy array of the shape (observations, ).
        params (dict): a configuration dictionary containing experiment hyperparamters.
            polynomial_order (int): The order of the polynomial library.
            number_of_freq (int): Number of Fourier frequencies in the library.
        
    Returns:
        (numpy array) current_observable: a 2-dimensional numpy array with the shape (num_of_trajectories * observations -1, dim. of observables).
        (numpy array) one_step_forward_observable: a 2-dimensional numpy array with the shape (num_of_trajectories * observations -1, dim. of observables).
        (numpy array) observation_time_diff: an numpy array with the shape (num_of_trajectories * observations -1, ).
        (list) feature_names: a list containing the names of observables.
    """

    libraries = []
    if params["polynomial_order"] > 0:
        libraries.append(ps.PolynomialLibrary(degree=params["polynomial_order"], include_bias=False))
    if params["number_of_freq"] > 0:
        libraries.append(ps.FourierLibrary(n_frequencies=params["number_of_freq"], include_sin=True, include_cos=True))

    if not libraries:
        raise RuntimeError("No library added.")
    observables_function = ps.ConcatLibrary(libraries)
    current_observable_list, one_step_forward_observable_list, time_diff_list = [], [], []
    
    for i in range(len(train_data)): # Iterate over the nubmer of trajectory
        temp_observable = observables_function.fit_transform(train_data[i])
        current_observable_list.append(temp_observable[:-1])
        one_step_forward_observable_list.append(temp_observable[1:])
        time_diff_list.append(np.diff(observation_time_train_data[i]))

    current_observable = np.concatenate(current_observable_list, axis=0)
    one_step_forward_observable = np.concatenate(one_step_forward_observable_list, axis=0)
    observation_time_diff = np.concatenate(time_diff_list, axis=0)

    return current_observable, one_step_forward_observable, observation_time_diff, observables_function.get_feature_names(["x_1","x_2"])

def evaluate(analysis, feature_names):
    """
    Process the raw optimization results into a list of optimal eigenpairs, one for each initial eigenvalue guess. 
    Args:
        analysis (dict): A dictionary where each key is an eigenvalue,
        and each value is a triplet representing an optimization history.
        feature_names (list): A list of observable names.
    Returns:
        (list) [perf_list, feature_names]: A list of the perf_list and the feature names.
        (list) perf_list: A list of a pair of the form (optimized eigenvalue, optimized eigenvector)
    """
    perf_list = []

    for eigenval_key in analysis:
        pred_eig_val = analysis[eigenval_key][0][-1] # the loss = the eigenvalue
        pred_coef = np.array(analysis[eigenval_key][-1]) # the optimized eigenvector
        perf_list.append([pred_eig_val, pred_coef])

    return [perf_list, feature_names]

def save_pickle(data, path):
    """
    Serialize and save a Python object to a file using the highest pickle protocol.

    Args:
        data (Any): The Python object to serialize. This can be any picklable Python datatype, 
                    such as dictionaries, lists, custom objects, etc.
        path (str): The file path where the serialized object will be saved.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)