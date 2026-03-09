import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from module.optics.JWST_simulation import intensity_JWST, jwst_pupil, get_true_blackbody_rgb

def add_gravitational_lens(sky_R, sky_G, sky_B, cx, cy, r_einstein, src_x, src_y, src_size, temp, brightness):
    """
    Simulates a gravitationally lensed background object.
    Randomly chooses between a simple star/elliptical galaxy OR a spiral galaxy!
    
    src_x (src_y): where the star/galaxy originally is
    """
    grid_size = int(r_einstein * 5)
    y, x = np.ogrid[-grid_size:grid_size, -grid_size:grid_size]
    r = np.sqrt(x**2 + y**2) + 1e-6
    
    # Backward Ray Tracing; lens equation: beta = theta * (1-(theat_E/theta)^2)
    shift_factor = 1.0 - (r_einstein**2 / r**2)
    mapped_x = x * shift_factor
    mapped_y = y * shift_factor
    
    # equal to 0 if mapped coordinate originates from original source
    dx = mapped_x - src_x
    dy = mapped_y - src_y
    
    # base Gaussian disk
    R = np.sqrt(dx**2 + dy**2) + 1e-6
    disk = np.exp(-0.5 * (R / src_size)**2)
    
    is_spiral = np.random.choice([True, False])
    
    if is_spiral:
        # Generate spiral arms
        theta = np.arctan2(dy, dx)
        num_arms = np.random.choice([2, 2, 3, 4])
        winding = np.random.uniform(1.0, 3.0)
        spin_dir = np.random.choice([-1, 1])
        arms = 0.5 + 0.5 * np.cos(num_arms * theta + spin_dir * winding * R)
        intensity = disk * (0.4 + 0.6 * arms)
    else:
        intensity = disk

    # Mask out the mathematical singularity at the dead center
    intensity[r < r_einstein * 0.2] = 0.0
    

    r_mult, g_mult, b_mult = get_true_blackbody_rgb(temp)
    
    y_min, y_max = max(0, cx - grid_size), min(sky_R.shape[0], cx + grid_size)
    x_min, x_max = max(0, cy - grid_size), min(sky_R.shape[1], cy + grid_size)
    
    gy_min = y_min - (cx - grid_size)
    gy_max = gy_min + (y_max - y_min)
    gx_min = x_min - (cy - grid_size)
    gx_max = gx_min + (x_max - x_min)
    
    local_intensity = intensity[gy_min:gy_max, gx_min:gx_max] * brightness
    
    sky_R[y_min:y_max, x_min:x_max] += local_intensity * r_mult
    sky_G[y_min:y_max, x_min:x_max] += local_intensity * g_mult
    sky_B[y_min:y_max, x_min:x_max] += local_intensity * b_mult


if __name__ == "__main__":
    # Simulate 3 wavelengths per color channel to smear the grating lobes
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
            raw_psf = intensity_JWST(1.0, 90.0, 0.0, wl, jwst_pupil, 
                                            focal_length=f_eff, screen_length=L, 
                                            n_grid=N, gpu=True, pad_factor=6)[0]
            
            # Crop the PSF from the center to remove any distant FFT edge ringing
            center = raw_psf.shape[0] // 2
            half_crop = psf_crop_size // 2
            cropped_psf = raw_psf[center-half_crop : center+half_crop, 
                                center-half_crop : center+half_crop]
            
            combined_psf += cropped_psf
            
        # Average the wavelengths and normalize
        combined_psf /= len(wls)
        psfs[color] = combined_psf / np.sum(combined_psf)

    height = 1080
    width = 1920
    sky_R = np.zeros((height, width))
    sky_G = np.zeros((height, width))
    sky_B = np.zeros((height, width))


    print("Generating starfield...")
    num_stars = 1500

    for _ in range(num_stars):
        ix = np.random.randint(0, height)
        iy = np.random.randint(0, width)
        
        brightness = 10 ** np.random.uniform(-3, 0)
        temp = np.random.uniform(3000, 20000) # Generate a random temperature (3000K Red Dwarf to 20000K Blue Giant)
        r_mult, g_mult, b_mult = get_true_blackbody_rgb(temp)

        sky_R[ix, iy] += brightness * r_mult
        sky_G[ix, iy] += brightness * g_mult
        sky_B[ix, iy] += brightness * b_mult

    print("Adding random gravitational lenses...")

    num_lenses = np.random.randint(50, 150)
    print(f"Injecting {num_lenses} gravitational lenses...")

    for _ in range(num_lenses):
        cx = np.random.randint(50, height - 50)
        cy = np.random.randint(50, width - 50)
        
        r_einstein = np.random.uniform(5.0, 18.0)
        
        src_x = np.random.uniform(-4.0, 4.0)
        src_y = np.random.uniform(-4.0, 4.0)
        
        src_size = np.random.uniform(0.2, 0.7)
        bg_temp = np.random.uniform(3000, 20000)
        bg_brightness = np.random.uniform(0.02, 0.08)
        
        add_gravitational_lens(sky_R, sky_G, sky_B, 
                            cx=cx, cy=cy, 
                            r_einstein=r_einstein, 
                            src_x=src_x, src_y=src_y, 
                            src_size=src_size, 
                            temp=bg_temp, brightness=bg_brightness)

        fg_brightness = np.random.uniform(0.5, 1.5)
        fg_temp = np.random.uniform(4000, 12000)
        fg_r, fg_g, fg_b = get_true_blackbody_rgb(fg_temp)
        
        sky_R[cx, cy] += fg_brightness * fg_r
        sky_G[cx, cy] += fg_brightness * fg_g
        sky_B[cx, cy] += fg_brightness * fg_b


    print("Taking 3-channel photo with JWST...")
    # Convolve each color sky with its PSF
    img_R = fftconvolve(sky_R, psfs['R'], mode='same')
    img_G = fftconvolve(sky_G, psfs['G'], mode='same')
    img_B = fftconvolve(sky_B, psfs['B'], mode='same')


    print("Processing RGB image...")
    final_img = np.dstack((img_R, img_G, img_B))

    # apply a global Logarithmic curve 
    final_img = np.log10(final_img + 1e-3) # add 1e-5 to prevent log(0) errors on the pitch-black pixels
    final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img)) # Normalize for Matplotlib


    fig, ax = plt.subplots(figsize=(16, 9), dpi=300, facecolor='black')
    ax.imshow(final_img, origin='lower')
    # ax.set_title(f"Simulated JWST Deep Field ({num_stars} Stars)", color='white', pad=15)
    ax.axis('off') 
    save_path = '../images/JWST/starfield_gravLense.png'
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300)
    print(f'{save_path} saved')
    plt.show()

    print()