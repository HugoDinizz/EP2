# Exercise 4 - Formulation C: inverse overdensity g = 1/delta and velocity divergence
import numpy as np




name = 'C'
# State y = [g, theta, delta_L, delta_L_N]
# growing mode: g_i = 1/delta_i, theta_i = -f_i delta_i / (1 + delta_i)
def initial_state(delta_i, f_i):
    return np.array([1 / delta_i, -f_i * delta_i / (1 + delta_i), delta_i, f_i * delta_i])

def rhs(N, y, cosmo):
    a = np.exp(N)
    g, th, dL, dLN = y
    AH, Om = cosmo.A_H(a), cosmo.Om(a)
    gN   = g * (1 + g) * th
    thN  = -AH * th - th**2 / 3 - 1.5 * Om / g # still singular as g -> 0
    dLNN = -AH * dLN + 1.5 * Om * dL # linear
    return [gN, thN, dLN, dLNN]

# Terminal event: collapse when g falls to epsilon > 0 (g decreases, so direction = -1)
def collapse_event(eps):
    def event(N, y, cosmo):
        return y[0] - eps
    event.terminal = True
    event.direction = -1
    return event

# Quantities from the state
def delta(y):
    return 1 / y[0] # delta = 1/g

def theta(y):
    return y[1]

def delta_L(y):
    return y[2]
