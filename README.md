# EP2 - Numerical Cosmology 

Solutions to the EP2 exercises. Each exercise has its own folder, and the figures produced by each one are saved in its `output/` subfolder.

| Folder | Exercise | Files |
|---|---|---|
| `E1/` | Linear growth | `exercise_1.py` (growth solver) + `exercise_1.ipynb` |
| `E2/` | Mode coupling | `exercise_2.ipynb` only |
| `E3/` | Gaussian field and Zel'dovich initial conditions | `exercise_3.py` (field, displacement, CIC, P(k)) + `exercise_3.ipynb` |
| `E4/` | Spherical collapse with three formulations | several modules + `run_all.py` + `exercise_4.ipynb`  |
| `E5/` | CLASS vs CAMB | `exercise_5.ipynb` only, with the CLASS input in `CLASS_config/` |

**Exercise 4** is the largest one and has more structure, as requested. The physics is split into modules
(`cosmology.py`, `form_A.py`, `form_B.py`, `form_C.py`, `collapse.py`), `run_all.py` reproduces every table and figure,
and the folder has its own README and a LaTeX report. The other exercises are notebooks, using a small `.py` file only when
its functions are reused elsewhere.

## Dependencies between exercises

- `E2` uses the Gaussian field generator and the P(k) estimator from `E3/exercise_3.py`.
- `E3` and `E4` use the growth solver from `E1/exercise_1.py` for D(a) and f(a).

The notebooks import these files through relative paths, so keep the folder structure.

## Environment

Everything was run in a conda environment. Versions used, for comparison:

| Package | Version |
|---|---|
| Python | 3.12.15 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| classy (CLASS) | 3.4.1 |
| camb | 2.0.4 |
| jupyterlab | 4.6.4 |
| ipykernel | 7.4.0 |


