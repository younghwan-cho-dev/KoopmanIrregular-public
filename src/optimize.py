import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

"""
Optimization utilities for Koopman eigenfunction learning.

This module provides core routines to construct data-driven loss functions and
optimize Koopman eigenfunctions via iterative minimization of an objective derived
from observable data.

Functions
---------
construct_loss_function(current_observable, one_step_forward_observable, observation_time_diff)
    Returns a callable loss function for a given Koopman eigenvalue candidate, based on observable mismatches.

partial_minimization_in_eigenfunction(input_matrix)
    Performs partial minimization by computing the smallest eigenvalue and corresponding eigenvector
    of a symmetric matrix constructed from data.

fit_koopman_eigenfunction(f0, INIT_VAl, params)
    Iteratively optimizes the Koopman eigenvalue and eigenfunction. Tracks convergence history of
    eigenvalue estimates and associated loss values.

Notes
-----
- The loss function approximates Koopman operator evolution error over one-step transitions in observable space.
- The optimization is scalar (1D over real-valued eigenvalues) and uses bounded line search.
- Eigenfunctions are recovered by minimizing the lowest eigenvalue of a constructed symmetric matrix.

Author: Younghwan Cho
"""


def construct_loss_function(
    current_observable, one_step_forward_observable, observation_time_diff
):
    """
    Returns a callable loss function for a given Koopman eigenvalue candidate, based on observable mismatches.

    Args:
        current_observable (numpy arrary): a 2-dimensional numpy array of the shape (num_of_data points-1, num_of_observables).
        one_step_forward_observable (numpy array): a 2-dimensional numpy array of the shape (num_of_data points-1, num_of_observables).
        observation_time_diff (numpy arrray): an numpy array of the shape (num_of_data points-1, ).

    Returns:
        (function) multivariate_function_D: A callable function that takes a Koopman eigenvalue candiate.
        and returns the Hessian of the objective function w.r.t. coefficients of a Koopman eigenfunction.
    """

    def multivariate_function_D(mu):

        c_i = one_step_forward_observable - np.exp(mu * np.atleast_2d(observation_time_diff).T) * current_observable
        return (1/ current_observable.shape[0]) * (c_i.T @ c_i)

    return multivariate_function_D

def partial_minimization_in_eigenfunction(input_matrix):

    """
    Performs partial minimization by computing the smallest eigenvalue and corresponding eigenvector
    of a symmetric matrix constructed from data.

    Arg:
        input matrix (numpy array): a 2-dimensional array of the shape (num_of_observables, num_of_observables).
        The return of a callable function multivariate_function_D given an candiate eigenvalue $\mu$.

    Returns:
        (float) fval: a loss that is equal to the smallest eigenvalue.
        (numpy array) opt_sol: a numpy array corresponding to the eigenvector of the smallest eigenvalue.
    """
    val, vec = eigh(input_matrix)
    val = np.clip(val.real,a_min=0,a_max=None)
    min_index = np.where(val == min(val))[0][0]
    opt_sol = vec[:, min_index].real
    fval = max(0, val[min_index].real)

    return fval, opt_sol

def fit_koopman_eigenfunction(f0, INIT_VAl, params):
    """
    Iteratively optimizes the Koopman eigenvalue and eigenfunction. Tracks convergence history of
    eigenvalue estimates and associated loss values.

    Args:
        f0 (function): a callable function that returns the hessian of the objective function given $\mu$.
        INIT_VAL (float): an initial guess on the Koopman eigenvalue.
        params (dict): a configuration dictionary containing experiment hyperparamters.

    Returns:
        (tuple) (eig_val_list, loss_list, optimal_eigenvector): a triplet of the optimization history and the resulting optimal eigenvector.
        (list) eig_val_list: a list of updated eigenvalues.
        (list) loss_list: a list of losses.
        (list) optimal_eigenvector: an optimal eigenvector or an eigenvector at the algorithm termination.
    """

    NUM_ITERATION= params["NUM_ITERATION"]
    LINE_SEARCH_BOUND = params["LINE_SEARCH_BOUND"]

    loss_list = []
    eig_val_list = []

    eigenvalue = INIT_VAl
    init_D_xy = f0(eigenvalue)
    init_optimization = partial_minimization_in_eigenfunction(init_D_xy)

    prev_loss = init_optimization[0]
    optimal_eigenvector = init_optimization[1]

    eig_val_list.append(eigenvalue)
    loss_list.append(init_optimization[0])

    def scalar_function_D(part):
        matrix_D = f0(part)
        loss, _ = partial_minimization_in_eigenfunction(matrix_D)
        return loss
    iter=1
    while iter < NUM_ITERATION and not np.isclose(prev_loss,0, atol=1.0E-20):
        real_part_optimization_result = minimize_scalar(
            lambda real_part: scalar_function_D(real_part),
            bounds=[eigenvalue - LINE_SEARCH_BOUND, eigenvalue + LINE_SEARCH_BOUND],
            method='bounded'#, options= {'xatol':1e-10}
        )
        eigenvalue = real_part_optimization_result.x

        D_xy = f0(eigenvalue)
        optimization_result = partial_minimization_in_eigenfunction(D_xy)

        optimal_eigenvector = optimization_result[1]
        cur_loss = optimization_result[0]

        eig_val_list.append(eigenvalue)
        loss_list.append(cur_loss)

        if np.isclose(cur_loss, prev_loss, atol=params["THRESHOLD"]):
            break
        
        prev_loss = cur_loss
        iter += 1

    return [eig_val_list,loss_list, optimal_eigenvector]
