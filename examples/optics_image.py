import sys
sys.path.append('../')
from module.optics.general_diffraction import intensity_diffraction
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# ---- physcis params ----
theta_deg = 0.0
phi_deg = 90.0
wavelength = 500e-9
screen_length = 50e-3
n_xy = 2**14
I0 = 1.0
disk_radius = 2e-3
half_display = 5e-3
z = 2

def aperture(X, Y, xp):
    R = xp.sqrt(X**2 + Y**2)
    return (R > disk_radius).astype(float)

dpi = 150

fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

I_fft_init, x, y = intensity_diffraction(
    I0=I0, phi=phi_deg, theta=theta_deg, wavelength=wavelength,
    z=z, aperture_func=aperture, 
    screen_length=screen_length, n_grid=n_xy, process_time=False,
    gpu=True
)


mask_x = np.abs(x) <= half_display
mask_y = np.abs(y) <= half_display
I_cropped_init = I_fft_init[np.ix_(mask_y, mask_x)]

im = ax.imshow(I_cropped_init,
               extent=(-half_display*1000, half_display*1000, -half_display*1000, half_display*1000),
               cmap='inferno', origin='lower', interpolation='bicubic')

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
cbar = fig.colorbar(im, label="Intensity")

plt.tight_layout()