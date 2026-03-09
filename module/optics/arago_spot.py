import numpy as np
from scipy.integrate import simpson
from scipy.special import j0
import time

"""
normal incident plane wave only
"""
def psi_hole(r,z,k,a,n_rp:int):
    r = np.atleast_1d(r)
    rp = np.linspace(0,a,n_rp)

    R  = r[:, None]
    RP = rp[None, :]
    phase_term = np.exp(1j * k * RP**2 / (2 * z))
    bessel_term = j0(k * R * RP / z)
    integrand = RP * phase_term * bessel_term
    integral = simpson(integrand, rp, axis=1)

    C = k * np.exp(1j * k * r**2 / (2 * z)) / (1j * z)
    return C * integral


def intensity_arago(r, z, wavelength, disk_radius, I0, n_rp=3000):
    k = 2*np.pi / wavelength
    a = disk_radius
    psi = 1 - psi_hole(r,z,k,a,n_rp=n_rp)
    return I0*np.abs(psi)**2

"""
generalized incident plane wave
"""
def psi_hole_2D(X, Y, z, k, a, kx, ky, n_rp=3000, n_batch=1):
    """
    small angle approximation: theta and phi!
    z is wave propagation direction. phi is wrt to x and theta wrt z
    """
    rp = np.linspace(0, a, n_rp)
    phase_rp = np.exp(1j * k * rp**2 / (2*z))

    psi_hole = np.zeros_like(X, dtype=np.complex128)
    
    for i in range(0,X.shape[0], n_batch):
        X_batch = X[i : i+n_batch]
        Y_batch = Y[i : i+n_batch]
        Q_batch = np.sqrt((-k*X_batch/z + kx)**2 + (-k*Y_batch/z + ky)**2)

        rp_3D = rp[None,None,:]
        bessel_term = j0(rp_3D*Q_batch[:,:,None])
        integrand = rp_3D * phase_rp[None,None,:] * bessel_term
        psi_hole[i:i+n_batch] = simpson(integrand, x=rp, axis=2)
    
    r2 = X**2 + Y**2
    prefactor = k/(1j*z) * np.exp(1j*k*r2/(2*z))

    return prefactor * psi_hole

def intensity_arago_2D(r_screen, n_xy, z, wavelength, disk_radius, I0,
                       theta=0.0, phi=0.0, n_rp=3000, n_batch=50,
                       process_time=False):
    """
    kx vec = k cos(phi) (-x̂)
    ky vec = k sin(phi) sin(theta) (-ŷ)
    kz vec = k sin(phi) cos(theta) (+ẑ)
    """

    x = np.linspace(-r_screen, r_screen, n_xy)
    y = np.linspace(-r_screen, r_screen, n_xy)
    X, Y = np.meshgrid(x, y)

    if process_time:
        start_time = time.time()

    theta = np.radians(theta)
    phi = np.radians(phi)
    k = 2*np.pi / wavelength
    a = disk_radius

    kx = -k*np.cos(phi) # the minus sign originates from defining k vec coming down from 2nd quadrant
    ky = -k*np.sin(phi)*np.sin(theta)

    psi_hole = psi_hole_2D(X, Y, z, k, a, kx, ky, n_rp, n_batch)
    paraxial_phase_shift = z * (kx**2 + ky**2) / (2*k)
    psi_bg = np.exp(1j * (kx*X + ky*Y - paraxial_phase_shift))
    psi_total = psi_bg - psi_hole

    if process_time:
        end_time = time.time()
        total_time = end_time - start_time
        mins = int(total_time // 60)
        secs = total_time % 60
        print(f"Time taken: {mins} mins {secs:.2f} seconds\n")

    return I0 * np.abs(psi_total)**2