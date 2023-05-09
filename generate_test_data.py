import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

# Domain dimensions and time arrays
ndim = 2
# Nt, Nx, Ny = 50, 50, 50
# Nt, Nx, Ny = 100, 17, 17
Nt, Nx, Ny = 100, 65, 65
# Nt, Nx, Ny = 50, 17, 17
# Nt, Nx, Ny = 50, 9, 9

tstart, tend = 0.0, 5
t = np.linspace(tstart, tend, Nt)

lx, ly, lz = 0.2, 0.2, 0.2
x = np.linspace(-lx/2, lx/2, Nx)
y = np.linspace(-ly/2, ly/2, Ny)
xx, yy = np.meshgrid(x, y, indexing='ij')
dV = (x[1] - x[0]) * (y[1] - y[0])

# Case 1


q0 = 1.0
omega = 13.0
d = 0.5*lz
sigma = 0.02 * min(lx, ly)


def rho_spat(X, Y, t):
    charge = q0*np.cos(omega*t)
    mu = 0.5*d
    R_positive = np.exp(- (X**2 + (Y-mu)**2)/(2*sigma**2))
    R_negative = np.exp(- (X**2 + (Y+mu)**2)/(2*sigma**2))
    charge_distr = charge*R_positive/(R_positive.sum()*dV)
    charge_distr -= charge*R_negative/(R_negative.sum()*dV)
    return charge_distr


# To me it feels like the charge is already conserved and thus no current right?
def J_spat(X, Y, t):
    return [np.zeros(X.shape), np.zeros(Y.shape)]


rho = np.zeros((Nt, Nx, Ny))
J = np.zeros((Nt, ndim, Nx, Ny))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, time)
    J[i] = J_spat(xx, yy, time)

print("Case 1:", rho.shape, J.shape)
np.savez("case1.npz", t=t, rho=rho, J=J, x=x, y=y)

V = 0
f = 4.0/(tend-tstart)
d = 0.4*lz
sigma = 0.05 * min(lx, ly)


def rho_spat(X, Y, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    R = np.exp(- (X**2 + (Y-mu)**2)/(2*sigma**2))
    #denom = R.sum() * dV
    denom = 1.0 #1np.pi*sigma**2
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
