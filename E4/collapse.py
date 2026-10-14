# Exercise 4 - Shared collapse code for the three formulations
# A formulation module F must provide: name, initial_state(delta_i, f_i), rhs(N, y, cosmo),
# collapse_event(threshold), delta(y), theta(y), delta_L(y).
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

DELTA_C_EDS = 3 / 20 * (12 * np.pi)**(2/3) # exact EdS collapse threshold, 1.686470...


# turnaround when theta crosses -3 (decreasing)
def turnaround_event(F):
    def event(N, y, cosmo):
        return F.theta(y) + 3.0
    event.terminal = False
    event.direction = -1
    return event


# One integration from a_i to a_end --- stopping at the collapse threshold
def integrate(F, cosmo, delta_i, a_i, a_end, threshold, rtol=1e-10, atol=1e-12):
    f_i = cosmo.f(a_i)[0]
    return solve_ivp(F.rhs, (np.log(a_i), np.log(a_end)), F.initial_state(delta_i, f_i),
                     args=(cosmo,), method='DOP853', rtol=rtol, atol=atol,
                     events=[F.collapse_event(threshold), turnaround_event(F)])


# Shooting -- tune delta_i so that the terminal event happens at a_c
def shoot(F, cosmo, a_c, a_i, threshold, rtol=1e-10, atol=1e-12):
    N_c, N_end = np.log(a_c), np.log(a_c) + 0.5

    def residual(delta_i):                       # N of collapse minus ln a_c (> 0: collapses too late)
        sol = integrate(F, cosmo, delta_i, a_i, np.exp(N_end), threshold, rtol, atol)
        N_ev = sol.t_events[0][0] if len(sol.t_events[0]) else N_end + 1.0
        return N_ev - N_c

    guess = DELTA_C_EDS * a_i / a_c              # linear EdS estimate
    lo, hi = 0.5 * guess, 2.0 * guess
    while residual(lo) < 0: lo /= 2              # widen until the bracket really contains the root
    while residual(hi) > 0: hi *= 2
    delta_i = brentq(residual, lo, hi, xtol=1e-14 * guess, rtol=1e-15, maxiter=200)
    return delta_i, (lo, hi)


# vvirial radius ratio y = R_vir/R_ta, root of 4 lam y^3 - 2(1+lam) y + 1 = 0 connected to 1/2 as lam --> 0
def virial_y(lam):
    if lam == 0:
        return 0.5
    roots = np.roots([4 * lam, 0.0, -2 * (1 + lam), 1.0])
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots > 0) & (roots < 1)]
    return roots[np.argmin(np.abs(roots - 0.5))]


# Full run for one formulation, cosmology and collapse epoch
def collapse(F, cosmo, a_c, a_i=1e-5, threshold=1e7, rtol=1e-10, atol=1e-12):
    delta_i, bracket = shoot(F, cosmo, a_c, a_i, threshold, rtol, atol)
    sol = integrate(F, cosmo, delta_i, a_i, a_c * np.exp(0.5), threshold, rtol, atol)

    N_ev, y_c = sol.t_events[0][0], sol.y_events[0][0]          # collapse
    N_ta, y_ta = sol.t_events[1][0], sol.y_events[1][0]         # turnaround
    a_ta, zeta = np.exp(N_ta), 1 + F.delta(y_ta)

    lam = cosmo.OL0 / (cosmo.Om0 * a_ta**-3 * zeta)             # rho_Lambda / rho_sph at turnaround
    y = virial_y(lam)
    Dvir_m = zeta * (a_c / a_ta)**3 * y**-3                     # formal-collapse-epoch convention

    return dict(formulation=F.name, cosmology=cosmo.name, a_c=a_c, a_i=a_i, threshold=threshold,
                rtol=rtol, atol=atol, delta_i=delta_i, bracket=bracket,
                event_residual=N_ev - np.log(a_c), delta_c=F.delta_L(y_c),
                a_ta=a_ta, zeta_ta=zeta, lambda_ta=lam, y=y,
                Dvir_m=Dvir_m, Dvir_c=cosmo.Om(a_c) * Dvir_m)