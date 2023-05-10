#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import speed_of_light, epsilon_0

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('-npz', type=str, required=True,
               help='''Numpy file with 4D charge density scalar rho(t,x,y,z),
               5D current density vector J(t,dim,x,y,z), time array t and
               grid coordinate arrays x, y, z''')
p.add_argument('-observation_points', type=float, nargs='+', required=True,
               help='N*3 coordinates of outside observation points')
args = p.parse_args()

npz_file = np.load(args.npz)
t = npz_file['t']
x, y, z = npz_file['x'], npz_file['y'], npz_file['z']
rho_samples = npz_file['rho']
J_samples = npz_file['J']
# Assume constant dx
dx = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
grid_size = rho_samples.shape[1:]
grid_coordinates = np.meshgrid(x, y, z, indexing='ij', sparse=True)

rho_deriv = np.gradient(rho_samples, t, axis=0)
J_deriv = np.gradient(J_samples, t, axis=0)
Jx_deriv, Jy_deriv, Jz_deriv = J_deriv[:, 0], J_deriv[:, 1], J_deriv[:, 2]

# RegularGridInterpolator is a bit overkill, since we do not interpolate in
# space, but the only way to vectorize time interpolation
drho_dt = RegularGridInterpolator((t, x, y, z), rho_deriv, bounds_error=False)
dJx_dt = RegularGridInterpolator((t, x, y, z), Jx_deriv, bounds_error=False)
dJy_dt = RegularGridInterpolator((t, x, y, z), Jy_deriv, bounds_error=False)
dJz_dt = RegularGridInterpolator((t, x, y, z), Jz_deriv, bounds_error=False)

dV = np.product(dx)
r_obs = np.reshape(args.observation_points, [-1, 3])
n_obs_points = len(r_obs)

# Compute delays per grid point for each observation point
delays = np.zeros((n_obs_points, *grid_size))

# Vectors r - r' for each grid point and each observation point
r_diff = np.zeros((n_obs_points, 3, *grid_size))

for i, r in enumerate(r_obs):
    for dim in range(3):
        r_diff[i, dim] = r[dim] - grid_coordinates[dim]
    delays[i] = norm(r_diff[i], axis=0)/speed_of_light

E_obs_rho = []
E_obs_J = []
coords = grid_coordinates
coords.insert(0, [])            # Dummy entry for time values

for i, r in enumerate(r_obs):
    # Observation time range
    n_obs_times = len(t)            # TODO: have more options
    t_obs = np.linspace(t.min()+delays[i].max(), t.max()+delays[i].min(),
                        n_obs_times)

    E_obs_rho.append(np.zeros((n_obs_times, 3)))
    E_obs_J.append(np.zeros((n_obs_times, 3)))

    # R is |r - r'| for each grid point
    R = delays[i] * speed_of_light
    factor = 1 / (4 * np.pi * epsilon_0 * speed_of_light * R)
    r_hat = r_diff[i] / R

    for k, t_o in enumerate(t_obs):
        # Time at source
        coords[0] = t_o - delays[i]
        coords_tuple = tuple(coords)

        # Interpolate at every grid point at the given t_source
        rho_term = drho_dt(coords_tuple)
        J_term = np.array([dJx_dt(coords_tuple), dJy_dt(coords_tuple),
                           dJz_dt(coords_tuple)]) / speed_of_light

        for dim in range(3):
            E_obs_rho[i][k, dim] = E_obs_rho[i][k, dim] + dV * \
                np.sum(factor * r_hat[dim] * rho_term)
            E_obs_J[i][k, dim] = E_obs_J[i][k, dim] - dV * \
                np.sum(factor * J_term[dim])

fig, ax = plt.subplots(n_obs_points, 3, sharex=True, sharey=True)

for i, r in enumerate(r_obs):
    ax[i, 0].plot(t_obs, E_obs_rho[i][:, 0], label='Ex')
    ax[i, 0].plot(t_obs, E_obs_rho[i][:, 1], label='Ey')
    ax[i, 0].plot(t_obs, E_obs_rho[i][:, 2], label='Ez')
    ax[i, 0].legend()
    ax[i, 0].set_title(f'Observer {i+1} at {r} (rho)')

    ax[i, 1].plot(t_obs, E_obs_J[i][:, 0], label='Ex')
    ax[i, 1].plot(t_obs, E_obs_J[i][:, 1], label='Ey')
    ax[i, 1].plot(t_obs, E_obs_J[i][:, 2], label='Ez')
    ax[i, 1].legend()
    ax[i, 1].set_title(f'Observer {i+1} at {r} (J)')

    ax[i, 2].plot(t_obs, E_obs_rho[i][:, 0] + E_obs_J[i][:, 0], label='Ex')
    ax[i, 2].plot(t_obs, E_obs_rho[i][:, 1] + E_obs_J[i][:, 1], label='Ey')
    ax[i, 2].plot(t_obs, E_obs_rho[i][:, 2] + E_obs_J[i][:, 2], label='Ez')
    ax[i, 2].plot(t_obs, norm(E_obs_rho[i] + E_obs_J[i], axis=1),
                  '--', label='||E||')
    ax[i, 2].legend()
    ax[i, 2].set_title(f'Observer {i+1} at {r} (total)')

plt.show()
