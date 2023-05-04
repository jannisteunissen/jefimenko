import numpy as np
from scipy.constants import speed_of_light

# Domain dimensions and time arrays
ndim = 3
Nt, Nx, Ny, Nz = 100, 23, 20, 25
lx, ly, lz = 1.0, 1.0, 1.0
x = np.linspace(0, lx, Nx)
y = np.linspace(0, ly, Ny)
z = np.linspace(0, lz, Nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

tstart, tend = 0.0, 2.0
t = np.linspace(tstart, tend, Nt+1)

# Case 1


def rho_spat(X, Y, Z, t):
    if t < 0.5*(tstart+tend):
        return np.zeros(X.shape)
    else:
        return np.ones(X.shape)


def J_spat(X, Y, Z, t):
    if t < 0.5*(tstart+tend):
        return [np.zeros(X.shape), np.zeros(Y.shape), np.zeros(Z.shape)]
    else:
        return [np.ones(X.shape), np.ones(Y.shape), np.ones(Z.shape)]


rho = []
J = []
for i, time in enumerate(t):
    rho.append(rho_spat(xx, yy, zz, time))
    J.append(J_spat(xx, yy, zz, time))


print("Case 1:", np.array(rho).shape, np.array(J).shape)

# Saving all the needed data
np.savez_compressed("case1.npz", t=t, rho=np.array(rho), J=np.array(J),
                    x=x, y=y, z=z)

V = 0
f = 2
d = 0.2*lz


def rho_spat(X, Y, Z, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    sigma = 0.2
    R = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    denom = 1.0  # 2*np.pi*sigma**2
    return R/denom


def J_spat(X, Y, Z, t):
    rho = rho_spat(X, Y, Z, t)
    vz = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    return [np.zeros(X.shape), np.zeros(X.shape), rho*vz]


rho = []
J = []
for i, time in enumerate(t):
    rho.append(rho_spat(xx, yy, zz, time))
    J.append(J_spat(xx, yy, zz, time))


print("Case 2:", np.array(rho).shape, np.array(J).shape)

# Saving all the needed data
np.savez_compressed("case2.npz", t=t, rho=np.array(rho), J=np.array(J),
                    x=x, y=y, z=z)

# Case 3


V = 0.1*speed_of_light
f = 2
d = 0.2*lz


def rho_spat(X, Y, Z, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    sigma = 0.2
    R = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    denom = 1.0  # 2*np.pi*sigma**2
    return R/denom


def J_spat(X, Y, Z, t):
    rho = rho_spat(X, Y, Z, t)
    vz = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    return [np.zeros(X.shape), np.zeros(X.shape), rho*vz]


rho = []
J = []
for i, time in enumerate(t):
    rho.append(rho_spat(xx, yy, zz, time))
    J.append(J_spat(xx, yy, zz, time))


print("Case 3:", np.array(rho).shape, np.array(J).shape)

# Saving all the needed data
np.savez_compressed("case3.npz", t=t, rho=np.array(rho), J=np.array(J),
                    x=x, y=y, z=z)
