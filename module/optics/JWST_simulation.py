import numpy as np
import time
import colour
try:
    import cupy as cp
except ImportError:
    cp = None

def psi_pupil(angle_x, angle_z, aperture_func, wavelength, L, N, xp):
    """
    kx vec = k cos(phi) (-x̂)
    ky vec = k sin(phi) sin(theta) (-ŷ)
    kz vec = k sin(phi) cos(theta) (+ẑ)
    """
    k = 2*xp.pi/wavelength
    phi = xp.radians(angle_x)
    theta = xp.radians(angle_z)

    kx0 = -k*xp.cos(phi)
    ky0 = -k*xp.sin(phi)*xp.sin(theta)

    sin_alpha = min(xp.sqrt(kx0**2 + ky0**2) / k, 1.0)
    alpha_user = xp.arcsin(sin_alpha)
    dx = L / N
    if (wavelength / (2 * dx)) > 1.0:
        alpha_max = xp.pi / 2 
    else:
        alpha_max = xp.arcsin(wavelength / (2 * dx))

    if alpha_user > 0.9 * alpha_max:
        print(f"WARNING: Angle close to or exceeding Nyquist limit. Auto-adjusting N from {N}...")

        target_angle = alpha_user / 0.9
        target_sin = xp.sin(target_angle) if target_angle < (xp.pi / 2) else 1.0
        N_min_required = (2 * L / wavelength) * target_sin
        N_new = 2 ** int(xp.ceil(xp.log2(N_min_required))) # Find the next power of 2 that satisfies the requirement
        N = max(N, N_new)
        print(f"--> N has been increased to {N} to prevent aliasing.")

    # coordinates of the primary mirror plane
    x = xp.linspace(-L/2, L/2, N)
    y = xp.linspace(-L/2, L/2, N)
    X,Y = xp.meshgrid(x,y, indexing='xy')

    aperture = aperture_func(X,Y,angle_x,angle_z,xp).astype(float)
    return aperture * xp.exp(1j * (kx0*X + ky0*Y))

def psi_tot(psi_pupil, xp, pad_factor=2):
    N = psi_pupil.shape[0]
    
    if pad_factor > 1:
        N_pad = int(N * pad_factor)
        
        padded_psi = xp.zeros((N_pad, N_pad), dtype=psi_pupil.dtype)
        start = (N_pad - N) // 2
        end = start + N
        padded_psi[start:end, start:end] = psi_pupil
    else:
        padded_psi = psi_pupil
        N_pad = N

    psi_final = xp.fft.fftshift(xp.fft.fft2(padded_psi))
    
    return psi_final, N_pad

def intensity_JWST(I0, phi, theta, wavelength, aperture_func, focal_length=131.4, screen_length=8.0, n_grid: int=1024,
                          process_time=False, gpu=False, return_to_cpu=True, pad_factor=2):

    """
    aperture_func must be defined for free space (X,Y), like a whole or a slit.
    If simulating an object (assuming 100% absorption), then define the free space outside the object.
    """
    if gpu and cp is not None:
        xp = cp
    else:
        xp = np
        if gpu and cp is None:
            print("Warning: CuPy not found. Falling back to CPU (NumPy).")

    if process_time:
        start_time = time.time()

    psi_init = psi_pupil(phi, theta, aperture_func, wavelength, screen_length, n_grid, xp)
    psi_final, N_padded = psi_tot(psi_init, xp, pad_factor=pad_factor)

    N_actual = psi_init.shape[0]
    dx = screen_length/N_actual
    f_fft = xp.fft.fftshift(xp.fft.fftfreq(N_padded, d=dx))
    x_focal = f_fft * wavelength * focal_length
    y_focal = f_fft * wavelength * focal_length

    if process_time:
        end_time = time.time()
        total_time = end_time - start_time
        mins = int(total_time // 60)
        secs = total_time % 60
        print(f"Time taken: {mins} mins {secs:.2f} seconds\n")

    intensity = (I0 * xp.abs(psi_final)**2, x_focal, y_focal)

    if return_to_cpu:
        return tuple(arr.get() if hasattr(arr, 'get') else arr for arr in intensity)
    
    return intensity

def jwst_pupil(X, Y, angle_x, angle_z, xp):
    """
    Generates a 2D transmission mask of the JWST primary mirror.
    Features:
    - Accurate 18-segment hexagonal geometry (flat-topped)
    - Asymmetrical spider struts to match real 8-pointed PSF
    - 3D parallax shifting of foreground structures based on incident angle
    - Sub-pixel analytical anti-aliasing using Signed Distance Functions (SDFs)
    
    X, Y : 2D grid arrays in physical meters
    angle_x, angle_z : Incident angles in degrees (matching your psi_ini)
    xp : numpy or cupy namespace
    """
    # Each segment is about 1.32m across, creating a ~6.5m mirror
    flat_to_flat = 1.32      
    gap = 0.01                
    pitch = flat_to_flat + gap 
    
    # Secondary mirror and support structure
    D_sec = 0.74              # Diameter of the central obscuration
    strut_width = 0.05        # Faintness of secondary spikes
    d_struts = 7.2            # Separates primary and secondary planes
    
    # Initialize a 100% absorbing (0) mask
    pupil = xp.zeros_like(X, dtype=float)
    
    # Calculate the physical width of a single pixel for anti-aliasing
    dx_pupil = xp.abs(X[0, 1] - X[0, 0]) # Assumes a square grid where dx == dy
    
    # -----------------------------------------------------------
    # The 18 Hexagonal Segments (Stationary Primary Plane)
    # -----------------------------------------------------------
    # vertical and angled spikes
    v1_x, v1_y = 0.0, pitch
    v2_x, v2_y = pitch * xp.sqrt(3) / 2, pitch / 2
    
    # Axial coordinates (q, r) define the grid position
    ring1 = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]
    ring2 = [(0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), 
             (0,-2), (1,-2), (2,-2), (2,-1), (2,0), (1,1)]
    segments = ring1 + ring2
    
    r_in = flat_to_flat / 2.0  # Apothem: center to flat edge
    
    for q, r in segments:
        # Center of this specific hexagon
        cx = q * v1_x + r * v2_x
        cy = q * v1_y + r * v2_y
        
        # Grid coordinates relative to segment center
        dx = X - cx
        dy = Y - cy
        
        # Distance checks update for flat-topped hexagons
        d1 = xp.abs(dy)
        d2 = xp.abs(xp.sqrt(3)/2 * dx + 0.5 * dy)
        d3 = xp.abs(xp.sqrt(3)/2 * dx - 0.5 * dy)
        
        # SDF: Maximum distance to any of the 3 axes
        d_max = xp.maximum(xp.maximum(d1, d2), d3)
        
        # Smooth the edge by mapping (apothem - d_max) to a range of [0, 1] over a width of exactly one pixel
        dist_to_edge = r_in - d_max
        hex_mask = xp.clip(dist_to_edge / dx_pupil + 0.5, 0.0, 1.0)
        
        # Combine segments using xp.maximum to handle tiny overlaps
        pupil = xp.maximum(pupil, hex_mask)

    # -----------------------------------------------------------
    # 3D Parallax Shift for Secondary Structures
    # -----------------------------------------------------------
    phi = xp.radians(angle_x)
    theta = xp.radians(angle_z)
    
    # Prevent division by zero at normal angles (90,0)
    denom = xp.sin(phi) * xp.cos(theta) + 1e-12
    shift_x = -d_struts * (xp.cos(phi) / denom)
    shift_y = -d_struts * xp.tan(theta)
    
    # Create the shifted grid for the foreground objects
    X_shifted = X - shift_x
    Y_shifted = Y - shift_y

    # -----------------------------------------------------------
    # Add the Central Obscuration (Secondary Mirror)
    # -----------------------------------------------------------
    R_grid = xp.sqrt(X_shifted**2 + Y_shifted**2)
    # SDF: (radius - D_sec/2) maps (0 inside, 1 outside) with a smooth edge
    sec_mask = xp.clip((R_grid - (D_sec / 2)) / dx_pupil + 0.5, 0.0, 1.0)
    pupil = pupil * sec_mask
    
    # -----------------------------------------------------------
    # Asymmetrical Spider Struts
    # -----------------------------------------------------------
    
    # Strut 1: The Top Strut (90 degrees) - This causes the two horizontal spikes
    d_x1 = xp.abs(X_shifted)
    in_x1 = xp.clip(((strut_width / 2) - d_x1) / dx_pupil + 0.5, 0.0, 1.0)
    in_y1 = xp.clip(Y_shifted / dx_pupil + 0.5, 0.0, 1.0)
    strut1_mask = 1.0 - (in_x1 * in_y1)
    pupil = pupil * strut1_mask
    
    # Strut 2: Bottom-Right
    ang2 = xp.radians(-60)
    X2 = X_shifted * xp.cos(ang2) + Y_shifted * xp.sin(ang2)
    Y2 = -X_shifted * xp.sin(ang2) + Y_shifted * xp.cos(ang2)
    
    d_y2 = xp.abs(Y2)
    in_y2 = xp.clip(((strut_width / 2) - d_y2) / dx_pupil + 0.5, 0.0, 1.0)
    in_x2 = xp.clip(X2 / dx_pupil + 0.5, 0.0, 1.0)
    strut2_mask = 1.0 - (in_y2 * in_x2)
    pupil = pupil * strut2_mask
    
    # Strut 3: Bottom-Left
    ang3 = xp.radians(240)
    X3 = X_shifted * xp.cos(ang3) + Y_shifted * xp.sin(ang3)
    Y3 = -X_shifted * xp.sin(ang3) + Y_shifted * xp.cos(ang3)
    
    d_y3 = xp.abs(Y3)
    in_y3 = xp.clip(((strut_width / 2) - d_y3) / dx_pupil + 0.5, 0.0, 1.0)
    in_x3 = xp.clip(X3 / dx_pupil + 0.5, 0.0, 1.0)
    strut3_mask = 1.0 - (in_y3 * in_x3)
    pupil = pupil * strut3_mask
    
    # Clip everything once to ensure transmission is strictly [0, 1]
    return xp.clip(pupil, 0.0, 1.0)


def get_true_blackbody_rgb(temp):
    """
    Calculates the exact sRGB color of a star at a given temperature.
    (Keep this strictly on the CPU/NumPy for efficiency and compatibility!)
    """
    shape = colour.SpectralShape(360, 780, 5)
    sd = colour.sd_blackbody(temp, shape)
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    XYZ = colour.sd_to_XYZ(sd, cmfs)
    RGB = colour.XYZ_to_sRGB(XYZ / np.max(XYZ))
    
    return np.clip(RGB, 0, 1)