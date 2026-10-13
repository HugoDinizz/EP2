# Cosmological functions of Exercise 1

import numpy as np
from scipy.integrate import solve_ivp

# Normalized squared Hubble function
def E2(a, Om0, OL0):
    Ok0 = 1 - Om0 - OL0
    return Om0 * a**-3 + Ok0 * a**-2 + OL0

# Matter density Om(a)
def Om_a(a, Om0, OL0):
    return Om0 * a**-3 / E2(a, Om0, OL0)

# dlnH/dN = (1/2) dlnE^2/dN
def dlnH_dN(a, Om0, OL0):
    Ok0 = 1 - Om0 - OL0
    return -(3 * Om0 * a**-3 + 2 * Ok0 * a**-2) / (2 * E2(a, Om0, OL0))

# Differential equation of linear growth
def ODEr(N, y, Om0, OL0):
    a = np.exp(N)
    G, G_N = y
    return [G_N, -(2 + dlnH_dN(a, Om0, OL0)) * G_N + 1.5 * Om_a(a, Om0, OL0) * G]

# Solving the ODE with Runge-Kutta (DOP853)
def growth_ode(Om0, OL0, ai=1e-3, rtol=1e-10, atol=1e-12, a_eval=None, method='DOP853'):
    sol = solve_ivp(ODEr, (np.log(ai), 0.0), [ai, ai], args=(Om0, OL0),
                    method=method, rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    G1 = sol.y[0, -1]
    a = np.geomspace(ai, 1, 200) if a_eval is None else np.asarray(a_eval)
    G, G_N = sol.sol(np.log(a))
    return a, G / G1 , G_N / G #a, D and f
