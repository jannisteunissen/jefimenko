import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Domain dimensions and time arrays
ndim = 3
# Nt, Nx, Ny, Nz = 50, 51, 51, 51
# Nt, Nx, Ny, Nz = 50, 17, 17, 17
Nt, Nx, Ny, Nz = 50, 33, 33, 33
# Nt, Nx, Ny, Nz = 200, 9, 9, 9
# Nt, Nx, Ny, Nz = 50, 9, 9, 9

tstart, tend = 0.0, 1e-6
t = np.linspace(tstart, tend, Nt)

lx, ly, lz = 0.2, 0.2, 0.2
x = np.linspace(-lx/2, lx/2, Nx)
y = np.linspace(-ly/2, ly/2, Ny)
z = np.linspace(-lz/2, lz/2, Nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
dV = dx * dy * dz


# Case 1
q0 = 1.0
omega = 2 * (2*np.pi/tend)
d = 0.5*lz
sigma = 0.05 * min(lx, ly)
mu = 0.5*d


def rho_spat(X, Y, Z, t):
    charge = q0*np.cos(omega*t)
    # R_positive = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    # R_negative = np.exp(- (X**2 + Y**2 + (Z+mu)**2)/(2*sigma**2))
    # charge_distr = charge*R_positive/(R_positive.sum()*dV)
    # charge_distr -= charge*R_negative/(R_negative.sum()*dV)
    charge_distr = np.zeros_like(X)
    zlo = Nz//4+1
    zhi = (3*Nz)//4+1
    charge_distr[Nx//2+1, Ny//2+1, zlo] = -charge/dV
    charge_distr[Nx//2+1, Ny//2+1, zhi] = +charge/dV
    return charge_distr


def J_spat(X, Y, Z, t):
    II = -q0*omega*np.sin(omega*t)
    J = II / (dx * dy)
    J_spat = np.zeros((3, *X.shape))
    zlo = Nz//4+1
    zhi = (3*Nz)//4+1
    J_spat[2, Nx//2+1, Ny//2+1, zlo:zhi+1] = J
    J_spat[2, Nx//2+1, Ny//2+1, zlo] = 0.5*J
    J_spat[2, Nx//2+1, Ny//2+1, zhi] = 0.5*J
    return J_spat


rho = np.zeros((Nt, Nx, Ny, Nz))
J = np.zeros((Nt, ndim, Nx, Ny, Nz))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, zz, time)
    J[i] = J_spat(xx, yy, zz, time)

print("Case 1:", rho.shape, J.shape)
np.savez("case1.npz", t=t, rho=rho, J=J, x=x, y=y, z=z)

V = 0
f = 2.0/(tend-tstart)
d = 0.2*lz
sigma = 0.2 * min(lx, ly, lz)


def rho_spat(X, Y, Z, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    R = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    denom = R.sum() * dV
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

print("Case 2:", rho.shape, J.shape)
np.savez("case2.npz", t=t, rho=rho, J=J, x=x, y=y, z=z)

# Case 3
V = 0 #3e-4*speed_of_light
# tstart, tend = 0.0, 0.2 * lz / V
f = 2.0/(tend-tstart)
d = 0.2*lz

t = np.linspace(tstart, tend, Nt)


def rho_spat(X, Y, Z, t):
    mu = V*t + d*np.sin(2*np.pi*f*t)
    sigma = 0.1 * min(lx, ly, lz)
    R = np.exp(- (X**2 + Y**2 + (Z-mu)**2)/(2*sigma**2))
    denom = R.sum() * dV
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


# Case 4
def particleToGrid(r, w):
    rho = np.zeros_like(xx)
    inv_dr = 1/np.array([dx, dy, dz])
    r0 = np.array([x[0], y[0], z[0]])

    tmp = (r - r0) * inv_dr
    l, m, n = tmp - np.floor(tmp)
    ix_lo = np.floor(tmp).astype(int)

    neighbors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    nearest_indices = neighbors + ix_lo

    weights = np.array([(1-l)*(1-m)*(1-n), l*(1-m)*(1-n), (1-l)*m*(1-n),
                        l*m*(1-n), (1-l)*(1-m)*n, l*(1-m)*n, (1-l)*m*n, l*m*n])

    for area_idx, idx in enumerate(nearest_indices):
        rho[idx[0], idx[1], idx[2]] = w * weights[area_idx] / dV

    return rho


V = 0 #3e-4*speed_of_light
# tstart, tend = 0.0, 0.2 * lz / V
f = 2.0/(tend-tstart)
d = 0.2*lz

# t = np.linspace(tstart, tend, Nt)


def rho_spat(X, Y, Z, t):
    xpos, ypos, zpos = 0, 0, V*t + d*np.sin(2*np.pi*f*t)
    rho = particleToGrid([xpos, ypos, zpos], 1.0)
    # plt.plot(rho[Nx//2, Ny//2, :])
    # plt.show()
    return rho


def J_spat(X, Y, Z, t):
    xpos, ypos, zpos = 0, 0, V*t + d*np.sin(2*np.pi*f*t)
    zvelocity = V + 2*np.pi*f*d*np.cos(2*np.pi*f*t)
    J = [np.zeros_like(X), np.zeros_like(X),
         particleToGrid([xpos, ypos, zpos], zvelocity)]
    return J


rho = np.zeros((Nt, Nx, Ny, Nz))
J = np.zeros((Nt, ndim, Nx, Ny, Nz))
for i, time in enumerate(t):
    rho[i] = rho_spat(xx, yy, zz, time)
    J[i] = J_spat(xx, yy, zz, time)

print("Case 4:", rho.shape, J.shape)
np.savez("case4.npz", t=t, rho=rho, J=J, x=x, y=y, z=z)
