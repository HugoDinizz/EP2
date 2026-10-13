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
# Apparently, the Nyquist freq. is negative in the fft method. As a consequence, the respective modes are self-conjugate. 

#This Nyquist scale is also related to the aliasing that happens to wavenumbers greater than k_Ny. 

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
