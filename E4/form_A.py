# Exercise 4 - Formulation A: second-order equation for delta, written as a first-order system
import numpy as np

name = 'A'
# State y = [delta, delta_N, delta_L, delta_L_N]
# growing mode: delta_N,i = f_i delta_i
def initial_state(delta_i, f_i):
    return np.array([delta_i, f_i * delta_i, delta_i, f_i * delta_i])

def rhs(N, y, cosmo):
    a = np.exp(N)
    d, dN, dL, dLN = y
    AH, Om = cosmo.A_H(a), cosmo.Om(a)
    dNN  = -AH * dN + (4/3) * dN**2 / (1 + d) + 1.5 * Om * d * (1 + d)    # nonlinear
    dLNN = -AH * dLN + 1.5 * Om * dL                                        # linear
    return [dN, dNN, dLN, dLNN]

# Terminal event: collapse when delta reaches the finite threshold delta_inf
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
    return -y[1] / (1 + y[0]) # theta = -delta_N / (1 + delta)

def delta_L(y):
    return y[2]