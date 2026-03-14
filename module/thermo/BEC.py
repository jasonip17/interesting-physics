import numpy as np
from mpmath import polylog
from scipy.optimize import brentq
from scipy.constants import hbar, k as kB

def N_BEC(N, m, V, n_points=200, z_min=1e-15, T_min=None, T_max=None):

    A = V * (m * kB / (2 * np.pi * hbar**2))**1.5
    Tc = (N / (A * float(polylog(1.5, 1.0).real)))**(2/3)
    if T_min == None:
        T_min = 0.1 * Tc
    if T_max == None:
        T_max = 2.0 * Tc
    T_array = np.linspace(T_min, T_max, n_points)

    N0_list = []
    N1_list = []
    z_list = []
    U_list = []

    for T in T_array:
        
        def N_particle_conservation(z):
            N0 = z / (1 - z)
            N1 = A * (T**1.5) * float(polylog(1.5, z).real)
            return (N0 + N1) - N
            
        z_max = N / (N + 1.0) # restrict N0 <= N
        z_exact = brentq(N_particle_conservation, z_min, z_max)
        
        N0_exact = z_exact / (1 - z_exact)
        N1_exact = A * (T**1.5) * float(polylog(1.5, z_exact).real)
        U_exact = 1.5 * kB * T * A * (T**1.5) * float(polylog(2.5, z_exact).real)
        
        N0_list.append(N0_exact)
        N1_list.append(N1_exact)
        z_list.append(z_exact)
        U_list.append(U_exact)

    N0_array = np.array(N0_list)
    N1_array = np.array(N1_list)
    U_array = np.array(U_list)

    Cv_array = np.gradient(U_array, T_array)
    

    return {
        'N0': N0_array,
        'N1': N1_array,
        'z': z_list,
        'T': T_array,
        'Tc': Tc,
        'U': U_array,
        'Cv': Cv_array
    }