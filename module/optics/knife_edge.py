import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import fresnel

def intensity_KnifeEdge(x, z, amplitude, wavelength, theta_degrees=0.0):
    """
    Fresnel Approximation
    amplitude is the background intensity I0
    theta_degrees in theta in degrees WRT horizontal
    """
    theta = np.radians(theta_degrees)
    x_shifted = x - z*np.sin(theta)

    lam = wavelength
    A = amplitude
    alpha = np.sqrt(2/(lam*z))
    u = alpha * x_shifted
    S, C = fresnel(u)

    return (A/2) * ((0.5 + C)**2 + (0.5 + S)**2)

def intensity_DoubleKnifeEdge(x, z, d, amplitude, wavelength, theta_degrees=0.0):
    theta = np.radians(theta_degrees)
    x_shifted = x - z*np.sin(theta)
    lam = wavelength
    amp = amplitude
    alpha = np.sqrt(2/(lam*z))
    u1 = alpha*(x_shifted-d/2)
    u2 = alpha*(x_shifted+d/2)

    S1,C1 = fresnel(u1)
    S2, C2 = fresnel(u2)

    psi = np.sqrt(amp/2) * ((1+1j)-((C2-C1) + 1j*(S2-S1)))

    return np.abs(psi)**2


def plot_KnifeEdge(x, z, amplitude, wavelength=633e-9, theta_degrees=0.0, clip=(1e-10, None)):
    I = intensity_KnifeEdge(x, z=z, amplitude=amplitude, wavelength=wavelength, theta_degrees=theta_degrees)
    I = np.clip(I, *clip)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    I_2D = np.tile(I, (100, 1))
    max_bright = np.max(I)

    ax1.imshow(I_2D, extent=[x.min(), x.max(), 0, 1], cmap='gist_heat', aspect='auto', norm=colors.LogNorm(vmin=clip[0], vmax=max_bright))
    ax1.set_yticks([])
    ax1.set_ylabel("Log Scaled")

    ax2.plot(x, I, color='rebeccapurple')
    ax2.set_ylabel("Intensity")
    ax2.set_xlabel("Screen Position (x)")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

def plot_DoubleKnifeEdge(x, z, d, amplitude, wavelength=633e-9, theta_degrees=0, clip=(1e-10, None)):
    I = intensity_DoubleKnifeEdge(x, z=z, d=d, amplitude=amplitude, wavelength=wavelength, theta_degrees=theta_degrees)
    I = np.clip(I, *clip)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    I_2D = np.tile(I, (100, 1))
    max_bright = np.max(I)

    ax1.imshow(I_2D, extent=[x.min(), x.max(), 0, 1], cmap='gist_heat', aspect='auto', norm=colors.LogNorm(vmin=clip[0], vmax=max_bright))
    ax1.set_yticks([])
    ax1.set_ylabel("Log Scaled")

    ax2.plot(x, I, color='rebeccapurple')
    ax2.set_ylabel("Intensity")
    ax2.set_xlabel("Screen Position (x)")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()
