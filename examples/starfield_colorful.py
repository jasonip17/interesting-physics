import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from module.optics.JWST_simulation import intensity_JWST, jwst_pupil, get_true_blackbody_rgb

wavelengths = {'R': 650e-9, 'G': 530e-9, 'B': 450e-9}
L = 8.0            
N = int(2**10)          
f_eff = 131.4

print("Calculating RGB PSFs...")
psfs = {}
for color, wl in wavelengths.items():
    intensity_data = intensity_JWST(1.0, 90.0, 0.0, wl, jwst_pupil, 
                                    focal_length=f_eff, screen_length=L, n_grid=N, gpu=True)
    psf = intensity_data[0]
    psfs[color] = psf / np.max(psf) # normalize

height = 1080
width = 1920
sky_R = np.zeros((height, width))
sky_G = np.zeros((height, width))
sky_B = np.zeros((height, width))

print("Generating starfield...")
num_stars = 400

for _ in range(num_stars):
    ix = np.random.randint(0, height)
    iy = np.random.randint(0, width)
    
    brightness = 10 ** np.random.uniform(-3, 0)
    temp = np.random.uniform(3000, 20000) 
    r_mult, g_mult, b_mult = get_true_blackbody_rgb(temp)
    sky_R[ix, iy] += brightness * r_mult
    sky_G[ix, iy] += brightness * g_mult
    sky_B[ix, iy] += brightness * b_mult

print("Taking 3-channel photo with JWST...")
img_R = fftconvolve(sky_R, psfs['R'], mode='same')
img_G = fftconvolve(sky_G, psfs['G'], mode='same')
img_B = fftconvolve(sky_B, psfs['B'], mode='same')

print("Processing RGB image...")
final_img = np.dstack((img_R, img_G, img_B))
luminance = np.sum(final_img, axis=2, keepdims=True) + 1e-10

# apply the Log stretch to the luminance
log_lum = np.log10(luminance + 1e-5)
log_lum_norm = (log_lum - np.min(log_lum)) / (np.max(log_lum) - np.min(log_lum))
final_color_img = (final_img / luminance) * log_lum_norm # Scale the original colors to match the new brightened luminance
final_color_img = np.clip(final_color_img, 0, 1)


fig, ax = plt.subplots(figsize=(16, 9), dpi=300, facecolor='black')
ax.imshow(final_color_img, origin='lower')
# ax.set_title(f"Simulated JWST Deep Field ({num_stars} Stars)", color='white', pad=15)
ax.axis('off') 
plt.tight_layout(pad=0)
save_path = '../images/JWST/starfield_color.png'
plt.savefig(save_path, dpi=300)
plt.show()
print()