import numpy as np
import camb
import matplotlib.pyplot as plt

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

plt.style.use('dark_background')


print("Running Einstein-Boltzmann solver (CAMB)...")
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06) # Set the standard Planck cosmological parameters
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)


results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
ell_1d = np.arange(len(totCL[:,0]))
Dl_TT = totCL[:,0]

# Convert D_l to C_l
Cl_TT = np.zeros_like(Dl_TT)
Cl_TT[2:] = Dl_TT[2:] * 2 * np.pi / (ell_1d[2:] * (ell_1d[2:] + 1))


N = 512
L = 10.0 # Size of the sky patch in degrees
dx = L / N

# Calculate frequencies in 'cycles per degree'
kx = np.fft.fftfreq(N, d=dx) 
ky = np.fft.fftfreq(N, d=dx)
kx_grid, ky_grid = np.meshgrid(kx, ky)

# Convert spatial frequency to spherical harmonic multipole 'ell'
ell_2d = np.sqrt(kx_grid**2 + ky_grid**2) * 360.0
Cl_2d = np.interp(ell_2d, ell_1d, Cl_TT, right=0)


print("Generating Quantum Fluctuations...")
noise_real = np.random.normal(0, 1, (N, N))
noise_imag = np.random.normal(0, 1, (N, N))
noise_fourier = noise_real + 1j * noise_imag

cmb_fourier = noise_fourier * np.sqrt(Cl_2d) # Multiply noise by sqrt of the True Power Spectrum

print("Expanding the Universe...")
cmb_map = np.real(np.fft.ifft2(cmb_fourier))
cmb_map = (cmb_map / np.std(cmb_map)) * 100 # Normalize the map to roughly the standard deviation of the true CMB (~100 microKelvin)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

im = ax1.imshow(cmb_map, cmap='RdYlBu_r', extent=[0, L, 0, L], origin='lower')
ax1.set_title("Simulated CMB", fontsize=14)
ax1.set_xlabel("Degrees")
ax1.set_ylabel("Degrees")
fig.colorbar(im, ax=ax1, label=r"$\Delta$ T ($\mu$K)", fraction=0.046, pad=0.04)

ax2.hist(cmb_map.flatten(), bins=100, histtype='step', density=True, color='gray', alpha=0.7)

x = np.linspace(np.min(cmb_map), np.max(cmb_map), 100)
gaussian_curve = (1 / (100 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / 100)**2)
ax2.plot(x, gaussian_curve, color='red', linewidth=2, label="Fit")

ax2.set_xlabel(r"Temperature Fluctuation ($\mu$K)")
ax2.set_ylabel("Probability Density")
ax2.legend()

fig.tight_layout()
save_path = '../images/JWST/cmb_gaussian.png'
plt.savefig(save_path, dpi=300)
print(f'{save_path} saved')
plt.show()