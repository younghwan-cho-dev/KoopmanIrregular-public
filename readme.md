# Koopman Operator Learning with Irregular Time Intervals

This repository contains code for learning Koopman eigenfunctions from dynamical systems, particularly under irregularly sampled time intervals. 

## Set Up Environment
We recommend using conda:
```
conda create -n koopman-env python=3.11 pip
conda activate koopman-env
pip install -r requirements.txt
```
## Run an Experiment
Run a predefined experiment from main.py:
```python main.py --task robustness``` or 
```python main.py --task high_order```

Next run eda.ipynb and robustness_test.ipynb to replicate the vizualizations and the numerical results.

## Exploring the Method
Modify configs/config.yaml to adjust:

- System type (Cubic_Nonlinear_2D_ODE, Duffing_System, Pendulum)
- Time interval and sampling style (uniform, random, irregular)
- Polynomial or Fourier observable order
- Number of trajectories and simulation span

and use run_experiment from experiment_runners.py to explore the method's capability.