import os
import logging
import numpy as np

from src.optimize import (
    construct_loss_function,
    partial_minimization_in_eigenfunction,
    fit_koopman_eigenfunction,
)
from src.dynamics import Cubic_Nonlinear_2D_ODE, Duffing_System, Pendulum
from src.simul import generate_synthetic_data
from src.utils import load_config, create_experiment_dir, preprocess_data, evaluate, save_pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def select_dynamics(dynamics_name: str):
    """
    Select dynamics of interests.

    Args:
        dynamics_name (str): a "dynamics" hyperparamters in the configuration .yml file.

    Returns:
        (function) dynamics: a callable function that defines a vector field for the initial value problem.
    """

    if dynamics_name == "Cubic_Nonlinear_2D_ODE":
        dynamics = Cubic_Nonlinear_2D_ODE
    elif dynamics_name == "Duffing_System":
        dynamics = Duffing_System
    elif dynamics_name == "Pendulum":
        dynamics = Pendulum
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_name}")

    return dynamics

def run_experiment(config: dict, experiment_dir: str):
    """
    Run a single experiment and store results. 

    Args:
        config (dict): a configuration dictionary containing experiment hyperparamters
        experiment_dir (str): Path to the directory where the experiment results will be saved.
    Returns:
        None
    """
    try:
        # Step 1: Generate synthetic data
        logger.info("Generating synthetic data...")
        dynamics = select_dynamics(config["params_synthetic_data"]["dynamics"])
        train_data, train_time = generate_synthetic_data(dynamics, config["params_synthetic_data"])

        # Step 2: Preprocess data
        logger.info("Preprocessing data...")
        X, Y, dt, feature_names = preprocess_data(train_data, train_time, config["params_preprocess"])

        # Step 3: Optimization
        logger.info("Running optimization...")
        loss_fn = construct_loss_function(X, Y, dt)

        # Landscape visualization
        eigenvalue_search_range = config["params_optimize"]["eigenvalue_search_range"]
        num_eigenvalue_grid_points = config["params_viz"]["num_eigenvalue_grid_points"]

        grid = np.linspace(*eigenvalue_search_range, num=num_eigenvalue_grid_points)
        losses = np.array([partial_minimization_in_eigenfunction(loss_fn(val))[0] for val in grid])
        save_pickle([grid, losses], os.path.join(experiment_dir, "landscape.pickle"))

        # Eigenvalue initialization
        num_initial_eigenvalue_guesses = config["params_optimize"]["num_initial_eigenvalue_guesses"]
        init_eigvals = np.linspace(*eigenvalue_search_range, num=num_initial_eigenvalue_guesses)

        # Koopman optimization
        analysis = {
            f"{round(val, 4)}": fit_koopman_eigenfunction(loss_fn, val, config["params_optimize"])
            for val in init_eigvals
        }
        save_pickle(analysis, os.path.join(experiment_dir, "analysis.pickle"))

        # Step 4: Evaluate
        logger.info("Evaluating results...")
        metrics = evaluate(analysis, feature_names)
        save_pickle(metrics, os.path.join(experiment_dir, "test_perf.pickle"))

        logger.info(f"Results saved in {experiment_dir}")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")

def robustness_test():
    """
    Run a robustness test to three different dynamical systems.
    The for loop changes the experiemnt configuration accordingly.
    """
    base_dir = create_experiment_dir("results_robustness")

    seed_list = [12]
    dynamics_list = ["Cubic_Nonlinear_2D_ODE", "Duffing_System", "Pendulum"]

    for dyn in dynamics_list:

        config = load_config("configs/config.yaml")

        if dyn == "Cubic_Nonlinear_2D_ODE":
            config["params_optimize"]["eigenvalue_search_range"] = [-3, 0]

        if dyn == "Duffing_System":
            config["params_preprocess"]["polynomial_order"] = 4
            config["params_viz"]["num_eigenvalue_grid_points"] = 100

        if dyn == "Pendulum":
            config["params_preprocess"]["polynomial_order"] = 4
            config["params_preprocess"]["number_of_freq"] = 1

        for seed in seed_list:
            config["params_synthetic_data"].update({"dynamics": dyn, "SEED": seed})
            dyn_short = dyn.split("_")[0]
            
            # Uniform Sampling
            for interval in [0.01]:
                config["params_synthetic_data"].update({"TIME_INTERVAL": interval, "randomness": False})
                exp_dir = os.path.join(base_dir, f"{dyn_short}_{seed}_{interval}_False")
                os.makedirs(exp_dir, exist_ok=True)
                run_experiment(config, exp_dir)

            # Random Stationary Sampling
            config["params_synthetic_data"].update({"TIME_INTERVAL": 0.1, "randomness": True})
            exp_dir = os.path.join(base_dir, f"{dyn_short}_{seed}_0.1_True")
            os.makedirs(exp_dir, exist_ok=True)
            run_experiment(config, exp_dir)

            # Irregular Sampling
            config["params_synthetic_data"].update({"TIME_INTERVAL": 0.1, "randomness": "Irregular"})
            exp_dir = os.path.join(base_dir, f"{dyn_short}_{seed}_0.1_Irregular")
            os.makedirs(exp_dir, exist_ok=True)
            run_experiment(config, exp_dir)

def discover_high_order():
    """
    Run a high-order Koopman eigenfunction identification experiment on Cubic Nonlinear ODE system
    """
    base_dir = create_experiment_dir("results_high_order")

    seed_list = [12]

    config = load_config("configs/config.yaml")

    config["params_synthetic_data"]["randomness"] = True
    config["params_preprocess"]["polynomial_order"] = 6

    config["params_optimize"]["eigenvalue_search_range"] = [-4.5, 0]
    config["params_optimize"]["num_initial_eigenvalue_guesses"] = 100

    config["params_viz"]["num_eigenvalue_grid_points"] = 800

    for seed in seed_list:
        config["params_synthetic_data"]["SEED"] = seed
        exp_dir = os.path.join(base_dir, f"Cubic_{seed}_0.01_True")
        os.makedirs(exp_dir, exist_ok=True)
        run_experiment(config, exp_dir)