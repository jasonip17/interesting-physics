import sys
sys.path.append('../')
from module.optics.general_diffraction import intensity_diffraction
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

plt.style.use('dark_background')

# ---- physcis params ----
theta_deg = 0.0
phi_deg = 90.0
wavelength = 500e-9
screen_length = 80e-3
n_xy = 2**13
I0 = 1.0
disk_radius = 2e-3
half_display = 5e-3

def aperture(X, Y, xp):
    R = xp.sqrt(X**2 + Y**2)
    return (R > disk_radius).astype(float)

# ---- animation controls ----
fps = 30
duration_s = 4.0
num_frames = int(duration_s * fps)
z_values = np.linspace(0.5, 2.0, num_frames)
dpi = 150
extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "16"]

fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

I_fft_init, x, y = intensity_diffraction(
    I0=I0, phi=phi_deg, theta=theta_deg, wavelength=wavelength,
    z=z_values[0], aperture_func=aperture, 
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
title_text = ax.set_title(f"z = {z_values[0]:.2f} m")

vmax = float(I_cropped_init.max())
im.set_clim(0, vmax)

fig.tight_layout()

_start_t = time.perf_counter()
_last_report_t = time.perf_counter()
_last_report_n = 0  # frames completed at last report
_last_msg_len = 0
def update(i):
    global _last_report_t, _last_report_n, _last_msg_len, _start_t
    z_current = float(z_values[i])

    I_fft, _, _ = intensity_diffraction(
        I0=I0, phi=phi_deg, theta=theta_deg, wavelength=wavelength,
        z=z_current, aperture_func=aperture,
        screen_length=screen_length, n_grid=n_xy, process_time=False, gpu=True
    )
    I_cropped = I_fft[np.ix_(mask_y, mask_x)]

    im.set_data(I_cropped)
    title_text.set_text(f"z = {z_current:.2f} m")

    now = time.perf_counter()
    dt = now - _last_report_t
    if dt >= 5.0:
        frames_done = (i + 1) - _last_report_n
        fps = frames_done / dt if frames_done > 0 else float("0.0")

        total_elapsed = now - _start_t
        el_mins, el_secs = divmod(int(total_elapsed), 60)
        elapsed_str = f"{el_mins:02d}m {el_secs:02d}s"

        frames_remaining = num_frames - (i + 1)
        if fps > 0:
            eta_seconds = frames_remaining / fps
            eta_mins, eta_secs = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_mins:02d}m {eta_secs:02d}s"
        else:
            eta_str = "--m --s"

        msg = f"Rendering {i+1}/{num_frames} | z={z_current:.3f}m | FPS: {fps:5.2f} | Elapsed: {elapsed_str} | ETA: {eta_str}"
        pad = " " * max(0, _last_msg_len - len(msg))  
        print("\r" + msg + pad, end="", flush=True)

        # Update trackers for the next 5-second window
        _last_msg_len = len(msg)
        _last_report_t = now
        _last_report_n = i + 1

    return (im, title_text)

ani = animation.FuncAnimation(
    fig, update, frames=num_frames,
    interval=1000/fps,
    blit=True
)

writer = animation.FFMpegWriter(
    fps=fps,
    codec="libx264",
    extra_args=extra_args
)

path = "videos/optics/arago_spot_z.mp4"
ani.save(path, writer=writer, dpi=dpi)
print(f"\nvideo saved in {path}")
print()
# save as a GIF:
# ani.save('z_diffraction.gif', writer='pillow', fps=10)