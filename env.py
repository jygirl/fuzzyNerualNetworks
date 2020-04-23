from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

def ym(case, t=0):
    if case == 1:
        return 0
    else:
        return np.sin(t)
        
def D(t):
    return 1 if np.sin(t) >= 0 else -1

def state(x, t, u, _):
    x1 = x[1]
    x2 = -0.1*x[1] - x[0]**3 + 12*np.cos(t) + u + D(t)
    return [x1,x2]

def error(e, t, e_wave, _):
    e1 = e[1] + 60 * e_wave
    e2 = -144*e[0] - 24*e[1] + 900 * e_wave
    return [e1, e2]

def state1(x, t, u, _):
    g , l = 9.8, 1
    mc, m = 1, 1
    d = (l*(4/3 - m*np.cos(x[0])**2/(mc+m)))
    x1 = x[1]
    x3 = (g*np.sin(x[0]) - (m*l*x[1]**2*np.cos(x[0])*np.sin(x[0]))/(mc+m))/d
    x4 = (np.cos(x[0])/(mc+m))/d
    x2 = x3 + x4*u + D(t)
    return [x1,x2]