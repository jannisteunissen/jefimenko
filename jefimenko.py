#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import speed_of_light, epsilon_0
from os.path import splitext

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('npz', type=str,
               help='''Numpy file with 4D charge density scalar rho(t,x,y,z),
               5D current density vector J(t,dim,x,y,z), time array t and
               grid coordinate arrays x, y, z''')
p.add_argument('-observation_points', type=float, nargs='+', required=True,
               help='N*3 coordinates of outside observation points')
args = p.parse_args()

file_prefix = splitext(args.npz)[0]
npz_file = np.load(args.npz)
t = npz_file['t']
x, y, z = npz_file['x'], npz_file['y'], npz_file['z']
rho_samples = npz_file['rho']
J_samples = npz_file['J']

# Assume constant dx
dx = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
dV = np.product(dx)
grid_size = rho_samples.shape[1:]
grid_coordinates = np.meshgrid(x, y, z, indexing='ij', sparse=True)

# Take time derivatives at center of time intervals. They are second order
# accurate, and as local as possible (compared to e.g. central differencing)
N_t = len(t) - 1
t_deriv = 0.5 * (t[:-1] + t[1:])
rho_deriv = np.zeros((N_t, *grid_size))
J_deriv = np.zeros((N_t, 3, *grid_size))

for i in range(len(t) - 1):
    inv_dt = 1/(t[i+1] - t[i])
    rho_deriv[i] = (rho_samples[i+1] - rho_samples[i]) * inv_dt
    J_deriv[i] = (J_samples[i+1] - J_samples[i]) * inv_dt


# RegularGridInterpolator is a bit overkill, since we do not interpolate in
# space, but the only way to vectorize time interpolation
drho_dt = RegularGridInterpolator((t_deriv, x, y, z), rho_deriv)
dJx_dt = RegularGridInterpolator((t_deriv, x, y, z), J_deriv[:, 0])
dJy_dt = RegularGridInterpolator((t_deriv, x, y, z), J_deriv[:, 1])
dJz_dt = RegularGridInterpolator((t_deriv, x, y, z), J_deriv[:, 2])

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
t_obs_array = []
coords = grid_coordinates
coords.insert(0, [])            # Dummy entry for time values

for i, r in enumerate(r_obs):
    # Observation time range
    small_time = 1e-5 * (t_deriv[1] - t_deriv[0])
    t_obs_min = t_deriv.min() + delays[i].max() + small_time
    t_obs_max = t_deriv.max() + delays[i].min() - small_time

    # Use same time step as input data
    dt_source = (t.max() - t.min()) / (len(t) - 1)
    n_obs_times = np.ceil((t_obs_max-t_obs_min)/dt_source).astype(int)

    if n_obs_times < 0.1 * len(t):
        raise ValueError(f"Not enough observation time for point {r}")

    t_obs = np.linspace(t_obs_min, t_obs_max, n_obs_times)
    t_obs_array.append(t_obs)

    E_obs_rho.append(np.zeros((n_obs_times, 3)))
    E_obs_J.append(np.zeros((n_obs_times, 3)))

    # R is |r - r'| for each grid point
    R = delays[i] * speed_of_light
    factor = 1 / (4 * np.pi * epsilon_0 * speed_of_light * R)
    r_hat = r_diff[i] / R

    for k, t_o in enumerate(tqdm(t_obs)):
        # Time at source
        coords[0] = t_o - delays[i]
        coords_tuple = tuple(coords)

        # Interpolate at every grid point at the given t_source
        rho_term = drho_dt(coords_tuple)
        J_term = np.array([dJx_dt(coords_tuple), dJy_dt(coords_tuple),
                           dJz_dt(coords_tuple)]) / speed_of_light

        for dim in range(3):
            E_obs_rho[i][k, dim] += dV * np.sum(factor * r_hat[dim] * rho_term)
            E_obs_J[i][k, dim] -= dV * np.sum(factor * J_term[dim])

    # Save to text file
    header = 'time E_rho_x E_rho_y E_rho_z E_J_x E_J_y E_J_z'
    fname = file_prefix + f'_observer_{r[0]}_{r[1]}_{r[2]}.txt'
    np.savetxt(fname,
               np.array([t_obs, E_obs_rho[i][:, 0],
                         E_obs_rho[i][:, 1], E_obs_rho[i][:, 2],
                         E_obs_J[i][:, 0], E_obs_J[i][:, 1],
                         E_obs_J[i][:, 2]]).T, header=header)
    print(f'Wrote {fname}')


fig, ax = plt.subplots(n_obs_points, 3, sharex='row', sharey=True)
if ax.ndim == 1:
    ax = ax[None, :]

for i, r in enumerate(r_obs):
    t_obs = t_obs_array[i]
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
    ax[i, 2].legend()
    ax[i, 2].set_title(f'Observer {i+1} at {r} (total)')

plt.show()
