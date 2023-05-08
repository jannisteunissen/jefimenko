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
dJz_dt = RegularGridInterpolator((t, x, y, z), Jy_deriv, bounds_error=False)

dV = np.product(dx)
r_obs = np.reshape(args.observation_points, [-1, 3])
n_obs_points = len(r_obs)

# Compute delays per grid point for each observation point
delays = np.zeros((n_obs_points, *grid_size))

for i, r in enumerate(r_obs):
    for dim in range(3):
        delays[i] = delays[i] + (r[dim] - grid_coordinates[dim])**2
    delays[i] = np.sqrt(delays[i])/speed_of_light

E_obs = []
coords = grid_coordinates
coords.insert(0, [])            # Dummy entry for time values

for i, r in enumerate(r_obs):
    # Observation time range
    n_obs_times = len(t)            # TODO: have more options
    t_obs = np.linspace(t.min()+delays[i].max(), t.max()+delays[i].min(),
                        n_obs_times)

    E_obs.append(np.zeros((n_obs_times, 3)))
    factor = 1 / (4 * np.pi * epsilon_0 * speed_of_light * norm(r_obs[i]))

    for k, t_o in enumerate(t_obs):
        coords[0] = t_o - delays[i]
        coords_tuple = tuple(coords)

        # Approximate unit vector from observation to source (since source
        # region is much smaller than norm(r))
        r_hat = -r / norm(r)

        # Interpolate at every grid point at the given t_source
        rho_term = drho_dt(coords_tuple)
        Jx_term = dJx_dt(coords_tuple) / speed_of_light
        Jy_term = dJy_dt(coords_tuple) / speed_of_light
        Jz_term = dJz_dt(coords_tuple) / speed_of_light

        E_obs[i][k, 0] = E_obs[i][k, 0] + dV * factor * \
            np.sum(r_hat[0] * rho_term - Jx_term)
        E_obs[i][k, 1] = E_obs[i][k, 1] + dV * factor * \
            np.sum(r_hat[1] * rho_term - Jy_term)
        E_obs[i][k, 2] = E_obs[i][k, 2] + dV * factor * \
            np.sum(r_hat[2] * rho_term - Jz_term)

# for i, r in enumerate(r_obs):
#     # Converting to spherical co-ords
#     r_sph = norm(r)
#     theta = np.arccos(r[2]/r_sph)
#     phi = np.arctan(r[1]/r[0])
#     print(r, [r_sph, theta, phi])


fig, ax = plt.subplots(n_obs_points, sharex=True)
if not hasattr(ax, 'size'):
    ax = [ax]

for i, r in enumerate(r_obs):
    ax[i].plot(t_obs, E_obs[i][:, 0], label='Ex')
    ax[i].plot(t_obs, E_obs[i][:, 1], label='Ey')
    ax[i].plot(t_obs, E_obs[i][:, 2], label='Ez')
    ax[i].plot(t_obs, norm(E_obs[i][:, :], axis=1), '--', label='||E||')
    ax[i].legend()
    ax[i].set_title(f'Observer {i+1} at {r}')

plt.show()
