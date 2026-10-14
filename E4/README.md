# Exercise 4 - Spherical collapse with three numerical formulations

Spherical collapse in EdS and flat ΛCDM (radiation neglected), solved with three independent formulations:
**A** (δ, δ_N), **B** (δ, ϑ) and **C** (g = 1/δ, ϑ). For each one, the initial amplitude is tuned so that the
collapse happens at the chosen epoch, and the code returns the collapse threshold δ_c, the turnaround and the
virial overdensities.

## Files

| File | Content |
|---|---|
| `cosmology.py` | Background (E², Ω_m(a), A_H) and the linear growth rate f from Exercise 1 |
| `form_A.py`, `form_B.py`, `form_C.py` | The three systems of equations, each integrated on its own |
| `collapse.py` | Shared steps: shooting for δ_i, turnaround, virial prescription |
| `run_all.py` | Runs every item and saves tables and figures to `output/` |
| `exercise_4.ipynb` | Same calculations, step by step, with figures for each formulation |

`cosmology.py` imports the growth solver from `../E1/exercise_1.py`, so keep the repository folder structure!!


## How to run

From inside the `E4/` folder:

```bash
python run_all.py  # tables and figures in output/
```

`run_all.py` takes about 20 s.

## Outputs (`output/`)

| File | Item |
|---|---|
| `e_eds_benchmarks.csv` | (e) EdS benchmarks at z_c = 0, 1, 2 |
| `f_cosmology_grid.csv`, `f_cosmology_grid.png` | (f) Ω_m0 = 0.2, 0.3, 0.4 and z_c = 0, 1, 2, with EdS baseline |
| `g_convergence.csv`, `g_convergence.png` | (g) changes in a_i, tolerances and collapse threshold |
| `g_cross_differences.csv` | (g) differences between A, B and C, and event residuals |

Each table row lists the tuned δ_i, the root-search bracket, the event residual and the solver settings.

## Fiducial settings

a_i = 10⁻⁵, δ_∞ = 10⁸ (ε = 10⁻⁸ for C), rtol = 10⁻¹⁰, atol = rtol · a_i · 10⁻².
