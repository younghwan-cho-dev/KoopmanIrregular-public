import math
import numpy as np
from scipy.stats import expon
from scipy.integrate import solve_ivp

"""
Simulation utilities for generating synthetic trajectory data from dynamical systems.

This module provides helper functions to:
- Sample initial conditions uniformly from the unit circle
- Simulate time steps using exponential or irregular intervals
- Generate synthetic time series data for user-specified dynamics

Functions
---------
sample_unit_circle(num_samples=10)
    Samples initial conditions uniformly from the unit circle in 2D.

exp_timestep(scale=0.01)
    Returns a random timestep drawn from an exponential distribution with the given scale.

random_observation(TIME_SPAN, scale)
    Generates observation times over a time span using exponentially distributed steps.

irregular_observation(periods, TIME_SPAN, tolerance=1e-8)
    Returns observation times aligned with integer multiples of multiple periods.

generate_synthetic_data(dynamics, params)
    Simulates multiple trajectories of a dynamical system under specified sampling schemes.

Notes
-----
- All random sampling uses NumPy's RNG seeded with the provided `params["SEED"]` to ensure reproducibility.
- The system dynamics must be specified as a callable compatible with `scipy.integrate.solve_ivp`.

Author: Younghwan Cho
"""

def sample_unit_circle(num_samples=10):
    # Generate random angles uniformly between 0 and 2*pi
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    
    # Calculate x and y coordinates on the unit circle
    x_samples = np.cos(angles)
    y_samples = np.sin(angles)
    
    # Combine x and y coordinates into an array of (x, y) points
    samples = np.column_stack((x_samples, y_samples))
    
    return samples

def exp_timestep(scale=0.01):
    """
    Exponential random variable with mean equals the scale.
    pdf = lambda * exp(-lambda * x) where lambda = 1/scale.
    Args:
        scale (float): Mean of the exponential distribution with lambda = 1/scale.
    Returns:
        (float) timestep: a random variable realized from the exponential distribution.
    """
    return expon.rvs(scale=scale)

def random_observation(TIME_SPAN, scale):
    t=0
    time_list = [0]
    
    while True:
        dt=exp_timestep(scale)
        t+=dt
        if t>TIME_SPAN:
            break
        time_list.append(t)
    return np.array(time_list)

def irregular_observation(periods, TIME_SPAN, tolerance=10**-8):

    periods = np.array(periods)
    time_list=[]
    t=0
    while True: 
        if (math.isclose(t % periods[0], 0, abs_tol=tolerance) or 
            math.isclose(t % periods[1], 0, abs_tol=tolerance) or 
            math.isclose(t % periods[2], 0, abs_tol=tolerance)):
            time_list.append(t)
        t+=1
        if t>(TIME_SPAN*100):
            break

    time_list = np.unique(sorted(time_list))/100
    return time_list


def generate_synthetic_data(dynamics, params):
    """
    Simulates multiple trajectories of a dynamical system under specified sampling schemes.
    Args:
        dynamics (function): a callable function that defines a vector field for the initial value problem.
        params (dict): a configuration dictionary containing experiment hyperparamters
            TIME_INTERVAL (float): Time interval for observations.
            TIME_SPAN (float): Total time span for simulation.
            NUM_TRAJECTORY (int): Number of trajectories to simulate.
            SEED (int): Seed for random number generation.
            randomness (bool | str): Whether to use random observation intervals.
            Uniform Sampling : False, Random Stationary Sampling: True, Irregular Sampling: "Irregular".
            integrator_keywords (dict): Additional arguments for the numerical ODE integrator.

    Returns: 
        (tuple) (train_data, train_time_data): a pair consists of train_data and train_time_data
        (list) train_data: a list with the length of NUM_TRAJECTORY. Each element of the list contains a 2-dimensional numpy array of the shape (observations, states).  
        (list) train_time_data: a list with the length of NUM_TRAJECTORY. Each element of the list contains a numpy array of the shape (observations, ).  
    """
    np.random.seed(params["SEED"]) # Initialize random seed

    initial_conditions = sample_unit_circle(params["NUM_TRAJECTORY"])
    train_data = []
    train_time_data = []
    for i in range(params["NUM_TRAJECTORY"]):
        if params["randomness"] == False:
            observation_time_train_data = np.arange(0, params["TIME_SPAN"], params["TIME_INTERVAL"])
        if params["randomness"] == True:
            observation_time_train_data = random_observation(TIME_SPAN=params["TIME_SPAN"], scale=params["TIME_INTERVAL"])
        if params["randomness"] == "Irregular":
            observation_time_train_data = irregular_observation(periods=[11,13,17], TIME_SPAN=params["TIME_SPAN"])

        train_time_data.append(observation_time_train_data)

        train_data.append(solve_ivp(
                dynamics, (observation_time_train_data[0], observation_time_train_data[-1]),
                initial_conditions[i], t_eval=observation_time_train_data, **(params["integrator_keywords"] or {})
            ).y.T)

    return train_data, train_time_data # [...,(observations, states),..] [...,(observations,),...]