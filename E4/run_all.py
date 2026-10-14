# Exercise 4 - Runs every item and saves tables and figures to output/
#   python run_all.py
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cosmology import Cosmology
from collapse import collapse, DELTA_C_EDS
import form_A, form_B, form_C

OUT = Path(__file__).resolve().parent / 'output'
FORMS = (form_A, form_B, form_C)

# Fiducial settings (best values of the grid in item g)
A_I, DELTA_INF, RTOL = 1e-5, 1e8, 1e-10

# EdS
EDS_TARGETS = dict(delta_c=DELTA_C_EDS, ata_ac=2**(-2/3), zeta_ta=(3*np.pi/4)**2, Dvir_m=18*np.pi**2)


# Same physical endpoint for every formulation -- delta = delta_inf for A, B and g = 1/delta_inf for C
def threshold(F, delta_inf):
    return 1 / delta_inf if F.name == 'C' else delta_inf




def run(F, cosmo, z_c, a_i=A_I, delta_inf=DELTA_INF, rtol=RTOL):
    r = collapse(F, cosmo, 1 / (1 + z_c), a_i=a_i, threshold=threshold(F, delta_inf),
                 rtol=rtol, atol=rtol * a_i * 1e-2)      # atol scaled to delta_i ~ a_i
    r.update(z_c=z_c, delta_inf=delta_inf, ata_ac=r['a_ta'] / r['a_c'],
             bracket_lo=r['bracket'][0], bracket_hi=r['bracket'][1])
    return r


# e) EdS benchmarks at several collapse redshifts
def eds_benchmarks(z_cs=(0, 1, 2)):
    return [run(F, Cosmology(1.0), z) for z in z_cs for F in FORMS]



# f) cosmology grid, with the EdS baseline
def cosmology_grid(Om0s=(1.0, 0.2, 0.3, 0.4), z_cs=(0, 1, 2)):
    return [run(F, Cosmology(Om0), z) for Om0 in Om0s for z in z_cs for F in FORMS]


# g) convergence: change one setting at a time (delta_i is retuned in every run)
def convergence(Om0=0.3, z_c=0, a_is=(1e-3, 1e-4, 1e-5), rtols=(1e-6, 1e-8, 1e-10, 1e-12),
                delta_infs=(1e5, 1e6, 1e7, 1e8)):
    cosmo, rows = Cosmology(Om0), []
    for F in FORMS:
        for a_i in a_is:
            rows.append(dict(run(F, cosmo, z_c, a_i=a_i), scan='a_i'))
        for rtol in rtols:
            rows.append(dict(run(F, cosmo, z_c, rtol=rtol), scan='rtol'))
        for d in delta_infs:
            rows.append(dict(run(F, cosmo, z_c, delta_inf=d), scan='delta_inf'))
    return rows


# Largest relative difference between A, B and C
def cross_differences(rows, keys=('delta_c', 'Dvir_m', 'ata_ac')):
    groups = {}
    for r in rows:
        case = (r['cosmology'], r['z_c'], r['a_i'], r['rtol'], r['delta_inf'])
        groups.setdefault(case, []).append(r)
    out = []
    for case, rs in groups.items():
        row = dict(zip(('cosmology', 'z_c', 'a_i', 'rtol', 'delta_inf'), case))
        for k in keys:
            row[f'rel_diff_{k}'] = np.ptp([r[k] for r in rs]) / abs(np.mean([r[k] for r in rs]))
        row['max_event_residual'] = max(abs(r['event_residual']) for r in rs)
        out.append(row)
    return out


COLUMNS = ['formulation', 'cosmology', 'z_c', 'a_i', 'rtol', 'atol', 'delta_inf', 'threshold', 'delta_i',
           'bracket_lo', 'bracket_hi', 'event_residual', 'delta_c', 'a_ta', 'ata_ac', 'zeta_ta', 'lambda_ta',
           'y', 'Dvir_m', 'Dvir_c']


def save_csv(rows, name, columns=None):
    columns = columns or list(rows[0].keys())
    with open(OUT / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


# f) delta_c(z_c), Delta_vir,m(z_c), Delta_vir,c(z_c) with EdS baselines
def plot_grid(rows, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for name in dict.fromkeys(r['cosmology'] for r in rows):
        rs = [r for r in rows if r['cosmology'] == name and r['formulation'] == 'A']
        z = [r['z_c'] for r in rs]
        style = dict(color='k', ls='--') if name == 'EdS' else dict(marker='o')
        for ax, key in zip(axes, ('delta_c', 'Dvir_m', 'Dvir_c')):
            ax.plot(z, [r[key] for r in rs], label=name, **style)
    for ax, lab in zip(axes, (r'$\delta_c$', r'$\Delta_{\mathrm{vir},m}$', r'$\Delta_{\mathrm{vir},c}$')):
        ax.set_xlabel(r'$z_c$'); ax.set_ylabel(lab); ax.set_xticks([0, 1, 2])
    axes[0].legend()
    fig.tight_layout()
    if save:
        fig.savefig(OUT / 'f_cosmology_grid.png', dpi=150)
    return fig


# g) delta_c against each numerical setting
def plot_convergence(rows, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    scans = (('a_i', r'$a_i$'), ('rtol', 'rtol'), ('delta_inf', r'$\delta_\infty$  (C: $\epsilon = 1/\delta_\infty$)'))
    for ax, (scan, lab) in zip(axes, scans):
        for F, mk in zip(FORMS, ('o', 's', '^')):
            rs = [r for r in rows if r['scan'] == scan and r['formulation'] == F.name]
            ax.semilogx([r[scan] for r in rs], [r['delta_c'] for r in rs], mk,
                        ms=8 if F.name == 'A' else 4, mfc='none' if F.name == 'A' else None, label=F.name)
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.set_xlabel(lab); ax.set_ylabel(r'$\delta_c$')
    axes[0].legend(title='formulation')
    fig.suptitle(r'Convergence, flat $\Lambda$CDM $\Omega_{m0} = 0.3$, $z_c = 0$')
    fig.tight_layout()
    if save:
        fig.savefig(OUT / 'g_convergence.png', dpi=150)
    return fig


def main():
    OUT.mkdir(exist_ok=True)

    eds = eds_benchmarks()
    save_csv(eds, 'e_eds_benchmarks.csv', COLUMNS)
    print('(e) EdS benchmarks  [targets: delta_c %.5f, a_ta/a_c %.5f, zeta %.4f, Dvir %.2f]' % tuple(EDS_TARGETS.values()))
    for r in eds:
        print(f"  {r['formulation']} z_c={r['z_c']}: delta_c={r['delta_c']:.5f} a_ta/a_c={r['ata_ac']:.5f} "
              f"zeta={r['zeta_ta']:.4f} Dvir_m={r['Dvir_m']:.2f}")

    grid = cosmology_grid()
    save_csv(grid, 'f_cosmology_grid.csv', COLUMNS)
    plot_grid(grid)
    print('(f) cosmology grid saved')

    conv = convergence()
    save_csv(conv, 'g_convergence.csv', ['scan'] + COLUMNS)
    save_csv(cross_differences(grid + conv), 'g_cross_differences.csv')
    plot_convergence(conv)
    print('(g) convergence saved')


if __name__ == '__main__':
    main()
