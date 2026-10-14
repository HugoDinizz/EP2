# Exercise 4 - Formulation B: density and velocity divergence
import numpy as np

name = 'B'
# State y = [delta, theta, delta_L, delta_L_N]
# growing mode: theta_i = -f_i delta_i / (1 + delta_i)
def initial_state(delta_i, f_i):
    return np.array([delta_i, -f_i * delta_i / (1 + delta_i), delta_i, f_i * delta_i])

def rhs(N, y, cosmo):
    a = np.exp(N)
    d, th, dL, dLN = y
    AH, Om = cosmo.A_H(a), cosmo.Om(a)
    dN   = -(1 + d) * th # continuity
    thN  = -AH * th - th**2 / 3 - 1.5 * Om * d # Euler + Poisson
    dLNN = -AH * dLN + 1.5 * Om * dL # linear
    return [dN, thN, dLN, dLNN]

# Terminal event -- collapse when delta reaches delta_inf
def collapse_event(delta_inf):
    def event(N, y, cosmo):
        return y[0] - delta_inf
    event.terminal = True
    event.direction = 1
    return event

# Quantities from the state
def delta(y):
    return y[0]

def theta(y):
    return y[1] # theta is a state variable here

def delta_L(y):
    return y[2]
