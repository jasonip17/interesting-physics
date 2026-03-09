import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def intensity_SingleSlit(x, amplitude, wavelength, slit_width, screen_distance, theta_deg=0.0):
    theta = np.radians(theta_deg)
    k = 2*np.pi / wavelength
    x_eff = x - screen_distance*np.sin(theta)
    arg = (k*slit_width/(2*screen_distance)) * x_eff
    envelope = slit_width * np.sinc(arg/np.pi)

    return np.abs(amplitude)**2 * envelope**2

def intensity_DoubleSlit(x, gamma, amplitude, wavelength, slit_width, slit_distance, screen_distance, phase=0.0):
    """
    gamma must be between 0 (observe individual photons) and 1 (no observation of individual photons)
    """
    A = amplitude
    l = wavelength
    a = slit_width
    d = slit_distance
    L = screen_distance
    phi = phase

    # wave number
    k = 2*np.pi/l
    
    # np.sinc is sin(pi x) / (pi x) in numpy
    wave1 = A*a*np.sinc(k*a/(2*L)*(x-0.5*(a+d)) / np.pi) # located at +x
    wave2 = A*a*np.sinc(k*a/(2*L)*(x+0.5*(a+d)) / np.pi) # located at -x
    cos = np.cos(k/L*(a+d)*x + phi)

    return 0.5*np.abs(A)**2 * a**2 * (wave1**2 + wave2**2 + 2*gamma*wave1*wave2*cos)


def plot1_DoubleSlit(x, gamma, params):
    I = intensity_DoubleSlit(x, gamma=gamma, **params)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    I_2D = np.tile(I, (100, 1))
    max_bright = np.max(I)

    ax1.imshow(I_2D, extent=[x.min(), x.max(), 0, 1], cmap='gist_heat', aspect='auto', norm=colors.LogNorm(vmin=1e-5, vmax=max_bright))
    ax1.set_title(fr"$\gamma$={gamma}")
    ax1.set_yticks([])

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



def plot2_DoubleSlitContrast(x, params, gamma1=1, gamma2=0):

    I_fringe = intensity_DoubleSlit(x, gamma=gamma1, **params)
    I_particle = intensity_DoubleSlit(x, gamma=gamma2, **params)


    fig, ((ax_top_left, ax_top_right), (ax_bot_left, ax_bot_right)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 6), sharex=True)

    max_bright = np.max(I_fringe)

    I_2D_fringe = np.tile(I_fringe, (100, 1))

    ax_top_left.imshow(I_2D_fringe, extent=[x.min(), x.max(), 0, 1], cmap='gist_heat', aspect='auto', norm=colors.LogNorm(vmin=1e-5, vmax=max_bright))
    ax_top_left.set_title(fr"$\gamma$ = {gamma1}")
    ax_top_left.set_yticks([])
    ax_top_left.set_ylabel("Log Scaled Wall")

    ax_bot_left.plot(x, I_fringe, color='rebeccapurple')
    ax_bot_left.set_ylabel("Intensity")
    ax_bot_left.set_xlabel("Screen Position (x)")
    ax_bot_left.axhline(0, color='black', linewidth=1)
    ax_bot_left.grid(True, alpha=0.5)
    ax_bot_left.set_xlim(np.min(x), np.max(x))
    ax_bot_left.set_ylim(bottom=0)



    I_2D_particle = np.tile(I_particle, (100, 1))

    ax_top_right.imshow(I_2D_particle, extent=[x.min(), x.max(), 0, 1], cmap='gist_heat', aspect='auto', norm=colors.LogNorm(vmin=1e-5, vmax=max_bright))
    ax_top_right.set_title(fr"$\gamma$ = {gamma2}")
    ax_top_right.set_yticks([])

    ax_bot_right.plot(x, I_particle, color='teal') # Made this one teal for contrast!
    ax_bot_right.set_xlabel("Screen Position (x)")
    ax_bot_right.axhline(0, color='black', linewidth=1)
    ax_bot_right.grid(True, alpha=0.5)
    ax_bot_right.set_xlim(np.min(x), np.max(x))
    ax_bot_right.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()