import numpy as np
from scipy.constants import speed_of_light

# Domain dimensions and time arrays
ndim = 3
# Nt, Nx, Ny, Nz = 200, 50, 50, 50
Nt, Nx, Ny, Nz = 50, 50, 50, 50

tstart, tend = 0.0, 1e-6
t = np.linspace(tstart, tend, Nt)

lx, ly, lz = 0.2, 0.2, 0.2
x = np.linspace(-lx/2, lx/2, Nx)
y = np.linspace(-ly/2, ly/2, Ny)
z = np.linspace(-lz/2, lz/2, Nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')


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


rho = np.zeros((Nt, Nx, Ny, Nz))
J = np.zeros((Nt, ndim, Nx, Ny, Nz))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, zz, time)
    J[i] = J_spat(xx, yy, zz, time)

print("Case 1:", rho.shape, J.shape)
np.savez("case1.npz", t=t, rho=rho, J=J, x=x, y=y, z=z)

V = 0
f = 2.0/(tend-tstart)
d = 0.025*lz
sigma = 0.2 * min(lx, ly, lz)


def rho_spat(X, Y, Z, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    R = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    denom = np.pi*sigma**2
    return R/denom


def J_spat(X, Y, Z, t):
    rho = rho_spat(X, Y, Z, t)
    vz = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    return [np.zeros(X.shape), np.zeros(X.shape), rho*vz]


rho = np.zeros((Nt, Nx, Ny, Nz))
J = np.zeros((Nt, ndim, Nx, Ny, Nz))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, zz, time)
    J[i] = J_spat(xx, yy, zz, time)

print(rho.sum(axis=1).sum(axis=1).sum(axis=1))

print("Case 2:", rho.shape, J.shape)
np.savez("case2.npz", t=t, rho=rho, J=J, x=x, y=y, z=z)

# Case 3
V = 0 #3e-4*speed_of_light
# tstart, tend = 0.0, 0.2 * lz / V
f = 2.0/(tend-tstart)
d = 0.2*lz
# print(2*np.pi*f*d/speed_of_light)
# print(tend)

t = np.linspace(tstart, tend, Nt)


def rho_spat(X, Y, Z, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    sigma = 0.1 * min(lx, ly, lz)
    R = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    denom = 2*np.pi*sigma**2
    return R/denom


def J_spat(X, Y, Z, t):
    rho = rho_spat(X, Y, Z, t)
    vz = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    return [np.zeros(X.shape), np.zeros(X.shape), rho*vz]


rho = np.zeros((Nt, Nx, Ny, Nz))
J = np.zeros((Nt, ndim, Nx, Ny, Nz))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, zz, time)
    J[i] = J_spat(xx, yy, zz, time)

print("Case 3:", rho.shape, J.shape)
np.savez("case3.npz", t=t, rho=rho, J=J, x=x, y=y, z=z)
