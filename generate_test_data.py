import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

# Domain dimensions and time arrays
ndim = 2
# Nt, Nx, Ny = 50, 50, 50
# Nt, Nx, Ny = 100, 17, 17
Nt, Nx, Ny = 50, 33, 33
# Nt, Nx, Ny = 50, 17, 17
# Nt, Nx, Ny = 50, 9, 9

tstart, tend = 0.0, 1e-6
t = np.linspace(tstart, tend, Nt)

lx, ly, lz = 0.2, 0.2, 0.2
x = np.linspace(-lx/2, lx/2, Nx)
y = np.linspace(-ly/2, ly/2, Ny)
xx, yy = np.meshgrid(x, y, indexing='ij')
dV = (x[1] - x[0]) * (y[1] - y[0])

# Case 1


def rho_spat(X, Y, t):
    if t < 0.5*(tstart+tend):
        return np.zeros(X.shape)
    else:
        return np.ones(X.shape)


def J_spat(X, Y, t):
    if t < 0.5*(tstart+tend):
        return [np.zeros(X.shape), np.zeros(Y.shape)]
    else:
        return [np.ones(X.shape), np.ones(Y.shape)]


rho = np.zeros((Nt, Nx, Ny))
J = np.zeros((Nt, ndim, Nx, Ny))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, time)
    J[i] = J_spat(xx, yy, time)

print("Case 1:", rho.shape, J.shape)
np.savez("case1.npz", t=t, rho=rho, J=J, x=x, y=y)

V = 0
f = 2.0/(tend-tstart)
d = 0.2*lz
sigma = 0.2 * min(lx, ly)


def rho_spat(X, Y, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    R = np.exp(- (X**2 + (Y-mu)**2)/(2*sigma**2))
    denom = R.sum() * dV
    return R/denom


def J_spat(X, Y, t):
    rho = rho_spat(X, Y, t)
    vz = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    return [np.zeros(X.shape), rho*vz]


rho = np.zeros((Nt, Nx, Ny))
J = np.zeros((Nt, ndim, Nx, Ny))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, time)
    J[i] = J_spat(xx, yy, time)

print("Case 2:", rho.shape, J.shape)
np.savez("case2.npz", t=t, rho=rho, J=J, x=x, y=y)

# Case 3
V = 0 #3e-4*speed_of_light
# tstart, tend = 0.0, 0.2 * lz / V
f = 2.0/(tend-tstart)
d = 0.2*lz

t = np.linspace(tstart, tend, Nt)


def rho_spat(X, Y, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    sigma = 0.1 * min(lx, ly)
    R = np.exp(- (X**2 + (Y-mu)**2)/(2*sigma**2))
    denom = R.sum() * dV
    return R/denom


def J_spat(X, Y, t):
    rho = rho_spat(X, Y, t)
    vz = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    return [np.zeros(X.shape), rho*vz]


rho = np.zeros((Nt, Nx, Ny))
J = np.zeros((Nt, ndim, Nx, Ny))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, time)
    J[i] = J_spat(xx, yy, time)

print("Case 3:", rho.shape, J.shape)
np.savez("case3.npz", t=t, rho=rho, J=J, x=x, y=y)
