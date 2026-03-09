import sys
sys.path.append('../')
from module.optics.JWST_simulation import jwst_pupil
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np 

L = 8.0 
N = 1024
x = xp.linspace(-L/2, L/2, N)
y = xp.linspace(-L/2, L/2, N)
X, Y = xp.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(7, 7))

# Define the circular path
alphas = xp.linspace(45, 45 + 360, 90)
max_tilt = 15.0 

phis = 90.0 + max_tilt * xp.cos(xp.radians(alphas))
thetas = -max_tilt * xp.sin(xp.radians(alphas))

phi_start = float(phis[0])
theta_start = float(thetas[0])

initial_pupil = jwst_pupil(X, Y, phi_start, theta_start, xp)
if hasattr(initial_pupil, 'get'):
    initial_pupil = initial_pupil.get()

im = ax.imshow(initial_pupil, cmap='gray', extent=[-L/2, L/2, -L/2, L/2], origin='lower')

ax.set_title(fr"JWST Pupil Parallax Shift: $\phi$ = {phi_start:.1f}°, $\theta$ = {theta_start:.1f}°")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.grid(color='white', alpha=0.2, linestyle='--')
fig.tight_layout()

def update(frame):
    current_theta = thetas[frame]
    current_phi = phis[frame]
    
    new_pupil = jwst_pupil(X, Y, current_phi, current_theta, xp)
    
    if hasattr(new_pupil, 'get'):
        new_pupil = new_pupil.get()

    im.set_data(new_pupil)
    
    # Cast current_theta to float just in case it's a CuPy scalar
    theta_val = float(current_theta)
    phi_val = float(current_phi)
    ax.set_title(rf"JWST Pupil Parallax Shift: $\phi$ = {phi_val:.1f}°, $\theta$ = {theta_val:.1f}°")
    
    return [im]

# 50ms per frame
ani = animation.FuncAnimation(fig, update, frames=len(thetas), interval=50, blit=True)

path = "../videos/optics/jwst_mirror.gif"
ani.save(path, writer='pillow')
print(f"\nvideo saved as {path}\n")