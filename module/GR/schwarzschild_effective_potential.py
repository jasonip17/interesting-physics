import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import M_sun, M_earth, G

# Newtonian effective potential
def Vn_eff(r,L):
    return -1/r + (L**2)/r**2
    # return -(G.value*M*m)/r + (L**2)/(2*m*r**2)

# Schwarzschild effective potential
def Vs_eff(r,L):
    # return 1-1/r + 1/r**2 - 1/r**3
    return np.sqrt(1-1/r + (L**2)/r**2 - (L**2)/r**3)
    # return np.sqrt(m**2 - (2*G.value*M*m**2)/r + (L**2)/r**2 - (2*G.value*M*L**2)/r**3)

M = M_sun.value
m = M_earth.value
r = np.linspace(1,10,1000)

plt.figure(figsize=(8,6))
# plt.plot(r,Vn_eff(r,L), label="Newtonian effective potential")
for L in range(0,21,5):
    plt.plot(r,Vs_eff(r,L), label=f"L={L}")

plt.title("Schwarzschild effective potential")
plt.ylim(0)
plt.xlabel("radius")
plt.ylabel(r"$V_{eff}(r)$, M=1, m=1, G=1")
plt.legend()
# plt.savefig("Schwarzschild_effective_potential.png", bbox_inches="tight")
plt.show()