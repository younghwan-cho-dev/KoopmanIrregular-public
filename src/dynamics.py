import numpy as np

def Cubic_Nonlinear_2D_ODE(t, x):
    return [-0.5 * x[0],
            -2*(x[1]-0.3 * x[0]**3)]

# Undamped Duffing system
def Duffing_System(t,x): 
    return [x[1],
            x[0]-x[0]**3]

# Undamped pendulum system
def Pendulum(t,x):
    return [x[1],
            -np.sin(x[0])]