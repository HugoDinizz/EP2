# Exercise 3 - Functions
import numpy as np

# k grid and FFT grid using numpy
def k_grid(N, L):
    k1 = 2*np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing='ij')
    return kx, ky, kz, np.sqrt(kx**2 + ky**2 + kz**2)

# Gaussian field with <|delta_k|^2> = V P(k), delta(x) = (1/V) sum_k delta_k e^{ikx} as stated in the exercise
def gaussian_field(N, L, Pk, seed):
    V = L**3
    rng = np.random.default_rng(seed)
    W = np.fft.fftn(rng.standard_normal((N, N, N)))
    *_, k = k_grid(N, L)
    P = np.zeros_like(k)
    P[k > 0] = Pk(k[k > 0])
    delta_k = np.sqrt(V * P) * W / np.sqrt(N**3)
    delta_k[0, 0, 0] = 0.0 # zero mode required
    delta_x = np.fft.ifftn(delta_k).real * N**3 / V # numpy ifftn has 1/N^3, convention of the exercise needs 1/V
    return delta_k, delta_x
# Apparently, the Nyquist freq. is negative in the fft method.
#This Nyquist scale is also related to the aliasing that happens to wavenumbers greater than k_N. 

#For exercise 2 and 3, we have to use a P(k) estimator.

#Binned power spectrum P(k) = 1/(N_k V) sum_{k in shell} |delta_k|^2
def power_spectrum(delta_k, L, kmax=None):
    N = delta_k.shape[0]
    V = L**3
    kf = 2 * np.pi / L # fundamental mode = shell width
    *_, k = k_grid(N, L)
    if kmax is None:
        kmax = np.pi * N / L # default: up to k_Nyq
    nbins = int(np.floor(kmax / kf - 0.5))
    edges = kf * (np.arange(nbins + 1) + 0.5) # shells [kf/2, 3kf/2), [3kf/2, 5kf/2), ...
    ibin = np.digitize(k.ravel(), edges) - 1 # shell index of each mode (-1 or nbins: outside)
    ok = (ibin >= 0) & (ibin < nbins)
    Nk   = np.bincount(ibin[ok], minlength=nbins)
    ksum = np.bincount(ibin[ok], weights=k.ravel()[ok], minlength=nbins)
    Psum = np.bincount(ibin[ok], weights=np.abs(delta_k.ravel()[ok])**2, minlength=nbins)
    good = Nk > 0
    return ksum[good] / Nk[good], Psum[good] / (Nk[good] * V), Nk[good] # mean k, P(k), N_k

# FFT conversions for delta(x) = (1/V) sum_k delta_k exp{ikx}
def to_x(dk, L):
    N = dk.shape[0]
    return np.fft.ifftn(dk).real * N**3 / L**3

def to_k(dx, L):
    N = dx.shape[0]
    return np.fft.fftn(dx) * L**3 / N**3

# Zero the three Nyquist planes
def zero_nyquist(delta_k):
    h = delta_k.shape[0] // 2
    out = delta_k.copy()
    out[h, :, :] = 0; out[:, h, :] = 0; out[:, :, h] = 0
    return out

# Zel'dovich displacement per unit growth, Psi_k = i k / k^2 delta_k  (without growth, multiply by D1 after this)
def za_displacement(delta_k, L):
    N = delta_k.shape[0]
    kx, ky, kz, k = k_grid(N, L)
    inv_k2 = np.zeros_like(k)
    inv_k2[k > 0] = 1 / k[k > 0]**2
    return np.array([to_x(1j * ki * inv_k2 * delta_k, L) for ki in (kx, ky, kz)])   # shape (3, N, N, N)

# Uniform particle load. One particle at the corner of each cell (Lagrangian coordinates q)
def particle_load(N, L):
    q1 = np.arange(N) * L / N
    return np.array(np.meshgrid(q1, q1, q1, indexing='ij'))                          # shape (3, N, N, N)

# Cloud-in-cell assignment of equal-mass particles onto an M^3 mesh
def cic_density(pos, M, L):
    x = pos.reshape(3, -1) / (L / M) # positions in cell units
    i0 = np.floor(x).astype(int)
    w1 = x - i0 # weight given to the next cell
    rho = np.zeros((M, M, M))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = np.where(dx, w1[0], 1 - w1[0]) * np.where(dy, w1[1], 1 - w1[1]) * np.where(dz, w1[2], 1 - w1[2])
                np.add.at(rho, ((i0[0] + dx) % M, (i0[1] + dy) % M, (i0[2] + dz) % M), w)
    return rho / rho.mean() - 1, rho.sum()

# CIC window in Fourier space (amplitude) - eq. 8.10 from the notes
def cic_window(M, L):
    kx, ky, kz, _ = k_grid(M, L)
    H = L / M / 2
    return (np.sinc(kx * H / np.pi) * np.sinc(ky * H / np.pi) * np.sinc(kz * H / np.pi))**2

