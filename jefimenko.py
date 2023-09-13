#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import argparse
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import speed_of_light, epsilon_0
from os.path import splitext
from multiprocessing import Pool

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('npz', type=str,
               help='''Numpy file with 4D charge density scalar rho(t,x,y,z),
               5D current density vector J(t,dim,x,y,z), time array t and
               grid coordinate arrays x, y, z''')
p.add_argument('-observation_points', type=float, nargs='+', required=True,
               help='N*3 coordinates of outside observation points')
p.add_argument('-np', type=int, default=4,
               help='Number of threads/cores to use')
args = p.parse_args()

file_prefix = splitext(args.npz)[0]
npz_file = np.load(args.npz)
t = npz_file['t']
x, y, z = npz_file['x'], npz_file['y'], npz_file['z']

# Assume constant dx
dx = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
dV = np.product(dx)
grid_coordinates = np.meshgrid(x, y, z, indexing='ij', sparse=True)

# Take time derivatives at center of time intervals. They are second order
# accurate, and as local as possible (compared to e.g. central differencing)
N_t = len(t) - 1
t_deriv = 0.5 * (t[:-1] + t[1:])

# Load all data
rho_samples = npz_file['rho']
J_samples = npz_file['J']
grid_size = rho_samples.shape[1:]

# One large array for derivatives
ddt_samples = np.zeros((N_t, *grid_size, 5))

# Compute time derivative
for i in range(N_t):
    inv_dt = 1/(t[i+1] - t[i])
    ddt_samples[i, :, :, :, 0] = (rho_samples[i+1] - rho_samples[i]) * inv_dt

    # Re-order J array
    tmp = (J_samples[i+1] - J_samples[i]) * inv_dt
    ddt_samples[i, :, :, :, 1] = tmp[0]
    ddt_samples[i, :, :, :, 2] = tmp[1]
    ddt_samples[i, :, :, :, 3] = tmp[2]
    ddt_samples[i, :, :, :, 4] = rho_samples[i]

# Save memory
del rho_samples, J_samples

# RegularGridInterpolator is a bit overkill, since we do not interpolate in
# space, but the only way to vectorize time interpolation
all_derivs = RegularGridInterpolator((t_deriv, x, y, z), ddt_samples)

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


def get_E_obs(t_obs, t_delay, grid_coordinates, r_hat, r_diff_norm, factor):
    coords_tuple = (t_obs-t_delay, *grid_coordinates)

    # Interpolate at every grid point at the given t_source
    tmp = all_derivs(coords_tuple)
    space_term = tmp[:, :, :, 4]
    rho_term = tmp[:, :, :, 0]
    J_term = tmp[:, :, :, 1:4] / speed_of_light
    E_obs_space, E_obs_rho, E_obs_J = np.zeros(3), np.zeros(3), np.zeros(3)

    for dim in range(3):
        E_obs_space[dim] = np.sum(factor * speed_of_light * r_hat[dim] *
                                  space_term * (1.0/r_diff_norm))
        E_obs_rho[dim] = np.sum(factor * r_hat[dim] * rho_term)
        E_obs_J[dim] = -np.sum(factor * J_term[:, :, :, dim])

    return E_obs_space, E_obs_rho, E_obs_J


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

    # R is |r - r'| for each grid point
    R = delays[i] * speed_of_light
    factor = dV / (4 * np.pi * epsilon_0 * speed_of_light * R)
    r_hat = r_diff[i] / R

    # Helper function
    def get_E_obs_for_point(t_obs):
        return get_E_obs(t_obs, delays[i], grid_coordinates, r_hat, R, factor)

    with Pool(args.np) as p:
        tmp = p.map(get_E_obs_for_point, t_obs, chunksize=1)
        E_obs_combined = np.array(tmp)
        E_obs_space, E_obs_rho, E_obs_J = E_obs_combined[:, 0], E_obs_combined[:, 1], E_obs_combined[:,2]

    # Save to csv file
    header = 't_obs,t_src,E_space_x,E_space_y,E_space_z, E_rho_x,E_rho_y,E_rho_z,E_J_x,E_J_y,E_J_z'
    fname = file_prefix + f'_observer_{r[0]}_{r[1]}_{r[2]}.csv'
    all_data = np.array([t_obs, t_obs-delays[i].mean(),
                         E_obs_space[:, 0], E_obs_space[:, 1], E_obs_space[:, 2],
                         E_obs_rho[:, 0], E_obs_rho[:, 1], E_obs_rho[:, 2],
                         E_obs_J[:, 0], E_obs_J[:, 1], E_obs_J[:, 2]]).T
    np.savetxt(fname, all_data, header=header, comments='', delimiter=',')
    print(f'Wrote {fname}')
