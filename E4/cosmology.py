# Exercise 4 - Background cosmology
import sys
from pathlib import Path
import numpy as np

# Reuse exercise 1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'E1'))
from exercise_1 import E2 as _E2, Om_a as _Om_a, dlnH_dN as _dlnH_dN, growth_ode


class Cosmology:

    def __init__(self, Om0):
        self.Om0 = Om0
        self.OL0 = 1.0 - Om0
        self.name = 'EdS' if Om0 == 1.0 else f'LCDM Om0={Om0}'

    def E2(self, a):
        return _E2(a, self.Om0, self.OL0)

    def Om(self, a):
        return _Om_a(a, self.Om0, self.OL0)

    # A_H = 2 + dlnH/dN, the friction term
    def A_H(self, a):
        return 2.0 + _dlnH_dN(a, self.Om0, self.OL0)

    # Linear growth rate f from the exercise 1 ODE
    def f(self, a, rtol=1e-11):
        a = np.atleast_1d(np.asarray(a, dtype=float))
        a_start = min(1e-8, a.min() * 1e-3) # start deep in matter domination
        _, _, f = growth_ode(self.Om0, self.OL0, ai=a_start, rtol=rtol,
                             atol=rtol * a_start * 1e-2, a_eval=a)
        return f