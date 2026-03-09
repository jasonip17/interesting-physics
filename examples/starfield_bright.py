import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from module.optics.JWST_simulation import intensity_JWST, jwst_pupil, get_true_blackbody_rgb

wavelengths = {
    'R': [630e-9, 650e-9, 670e-9], 
    'G': [510e-9, 530e-9, 550e-9], 
    'B': [430e-9, 450e-9, 470e-9]
}
L = 8.0            
N = int(2**12)          
f_eff = 131.4
psf_crop_size = 1024

print("Calculating Broadband RGB PSFs...")
psfs = {}
for color, wls in wavelengths.items():
    combined_psf = np.zeros((psf_crop_size, psf_crop_size))
    
    for wl in wls:
        intensity_data = intensity_JWST(1.0, 90.0, 0.0, wl, jwst_pupil, 
                                        focal_length=f_eff, screen_length=L, 
                                        n_grid=N, gpu=True, pad_factor=6)
        raw_psf = intensity_data[0]
        center = raw_psf.shape[0] // 2
        half_crop = psf_crop_size // 2
        cropped_psf = raw_psf[center-half_crop : center+half_crop, 
                              center-half_crop : center+half_crop]
        
        combined_psf += cropped_psf
        
    combined_psf /= len(wls)
    psfs[color] = combined_psf / np.sum(combined_psf)


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
final_img = np.log10(final_img + 1e-3)
final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img))

fig, ax = plt.subplots(figsize=(16, 9), dpi=300, facecolor='black')

ax.imshow(final_img, origin='lower')
# ax.set_title(f"Simulated JWST Deep Field ({num_stars} Stars)", color='white', pad=15)
ax.axis('off') 
save_path = '../images/JWST/starfield_bright.png'
plt.tight_layout(pad=0)
plt.savefig(save_path, dpi=300)
print(f'{save_path} saved')
plt.show()

print()