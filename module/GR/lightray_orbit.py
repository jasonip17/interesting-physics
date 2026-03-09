import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def W_eff(r, M):
    return (1 / r**2) * (1 - 2 * M / r)

def photon_geodesic(phi, y, M):
    """
    u'' + u = 3Mu^2

    u = 1/r
    y1 = u; dy1/dphi = y2
    y2 = du/dphi; dy2/dphi = 3My1^2 - y1
    """
    y1, y2 = y
    return [y2, 3*M*y1**2 -y1]

def path(b, M, r0=1e4, phi_max=8*np.pi, num_points=200, rtol=1e-10, atol=1e-13):
    u0 = 1/r0
    y1_init = u0
    y2_init = np.sqrt(1/b**2 - u0**2 + 2*M*u0**3) # incoming photon du/dphi>0 initially since -1/r^2 dr/dphi > 0 initially

    def event_horizon(phi, y, M):
        return y[0] - 1 / (2*M)
    event_horizon.terminal = True # stop integration if path hits event horizon (event_horizon=0)

    def escape(phi, y, M):
        return y[0] - u0
    escape.terminal = True
    escape.direction = -1 # Only stop on the way out (when u is decreasing), otherwise it stops at the beginning u0

    sol = solve_ivp(photon_geodesic, (0, phi_max), (y1_init, y2_init), args=(M,),
                    dense_output=True, events=[event_horizon, escape],
                    rtol=rtol, atol=atol)
    
    phi_vals = np.linspace(0, sol.t[-1], num_points)
    u_vals = sol.sol(phi_vals)[0] # only get u, [1] is u'
    r_vals = 1/u_vals
    x = r_vals * np.cos(-phi_vals) # -phi to flip path to go from right to up (instead of down)
    y = r_vals * np.sin(-phi_vals)
    
    return x, y

def find_boomerang_b(M, target_angle=2*np.pi, r0=1e4,
                     phi_max=100*np.pi, b_max_factor=50.0,
                     rtol=1e-10, atol=1e-13):
    if target_angle <= np.pi:
        raise ValueError("A photon cannot sweep an angle less than or equal to π and escape!")
    
    def final_angle(b, M, r0=r0):
        u0 = 1/r0
        y1_init = u0
        y2_init = np.sqrt(1/b**2 - u0**2 + 2*M*u0**3) 

        def escape(phi, y, M):
            return y[0] - u0
        escape.terminal = True
        escape.direction = -1 

        sol = solve_ivp(photon_geodesic, (0, phi_max), (y1_init, y2_init), args=(M,),
                        events=[escape], rtol=rtol, atol=atol)
        return sol.t[-1]
    
    print(f"Hunting for an impact parameter that yields {target_angle/np.pi:.2f}π...")
    def f(b):
        return final_angle(b, M) - target_angle

    b_min = np.sqrt(27)*M + 1e-11
    b_max = b_max_factor * M

    return brentq(f, b_min, b_max)