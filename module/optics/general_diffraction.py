import numpy as np
import time
try:
    import cupy as cp
except ImportError:
    cp = None

def psi_ini(angle_x, angle_z, aperture_func, wavelength, L, N, xp):
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
        
        # We need: alpha_user <= 0.9 * arcsin(wavelength * N_new / (2 * L))
        # Rearranging for N_new: N_new >= (2 * L / wavelength) * sin(alpha_user / 0.9)
        target_angle = alpha_user / 0.9
        target_sin = xp.sin(target_angle) if target_angle < (xp.pi / 2) else 1.0
        N_min_required = (2 * L / wavelength) * target_sin
        N_new = 2 ** int(xp.ceil(xp.log2(N_min_required))) # Find the next power of 2 that satisfies the requirement
        N = max(N, N_new)
        print(f"--> N has been increased to {N} to prevent aliasing.")

    x = xp.linspace(-L/2, L/2, N)
    y = xp.linspace(-L/2, L/2, N)
    X,Y = xp.meshgrid(x,y, indexing='xy')

    aperture = aperture_func(X,Y,xp).astype(float)
    return aperture * xp.exp(1j * (kx0*X + ky0*Y)), x, y

def psi_tot(psi0, wavelength, z, dx, xp):
    """
    Exact Helmholtz propagation using angular spectrum method.
    
    U0 : field at z=0 (2D complex array)
    wavelength : wavelength
    z : propagation distance
    dx : pixel spacing (assume square pixels)
    """

    N = psi0.shape[0]
    k = 2*xp.pi / wavelength

    # frequency domain
    fx = xp.fft.fftfreq(N, d=dx)
    fy = xp.fft.fftfreq(N, d=dx)
    FX, FY = xp.meshgrid(fx, fy)

    kx = 2*xp.pi*FX
    ky = 2*xp.pi*FY
    kz = xp.sqrt(k**2 - kx**2 - ky**2 + 0j)

    psi0_fft = xp.fft.fft2(psi0)
    psi0_fft_prop = psi0_fft * xp.exp(1j * z * kz)

    psi_final = xp.fft.ifft2(psi0_fft_prop)

    return psi_final

def intensity_diffraction(I0, phi, theta, wavelength, z, aperture_func, screen_length=10e-3, n_grid: int=1024,
                          process_time=False, gpu=False, return_to_cpu=True):

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

    psi0, x, y = psi_ini(phi, theta, aperture_func, wavelength, screen_length, n_grid, xp)
    dx = screen_length/n_grid
    psi = psi_tot(psi0, wavelength, z, dx, xp)

    if process_time:
        end_time = time.time()
        total_time = end_time - start_time
        mins = int(total_time // 60)
        secs = total_time % 60
        print(f"Time taken: {mins} mins {secs:.2f} seconds\n")

    intensity = (I0 * xp.abs(psi)**2, x, y)

    if return_to_cpu:
        return tuple(arr.get() if hasattr(arr, 'get') else arr for arr in intensity)
    
    return intensity