#!/usr/bin/env python3

import numpy as np
from np.linalg import norm
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light, epsilon_0

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('-rho', type=str, required=True,
               help='Numpy file with 4D charge density scalar (t,x,y,z)')
p.add_argument('-J', type=str, required=True,
               help='Numpy file with 5D current density vector (t,dim,x,y,z)')
p.add_argument('-t', type=str, required=True,
               help='Numpy file with times for rho and J')
p.add_argument('-dx', type=float, nargs=3, required=True,
               help='Grid spacing of rho and J data (their origin is 0)')
p.add_argument('-observation_points', type=float, nargs='+', required=True,
               help='N*3 coordinates of outside observation points')
args = p.parse_args()

t = np.load(args.t)
rho_samples = np.load(args.rho)
J_samples = np.load(args.J)
grid_size = rho_samples.shape[1:]

rho_deriv = np.gradient(rho_samples, t, axis=0)
J_deriv = np.gradient(J_samples, t, axis=0)

# TODO: use RegularGridInterpolator (a bit overkill, since we do not interpolate
# in space, but the only way to vectorize time interpolation?)
drho_dt = interp1d(t, rho_deriv, axis=0)

# TODO: split J into components?
dJ_dt = interp1d(t, J_deriv, axis=0)

domain_size = args.dx * rho_samples.shape[1:]
dV = np.product(args.dx)
r_obs = np.reshape(args.observation_points, [-1, 3])
n_obs_points = len(r_obs)

# Compute delays per grid point for each observation point
grid_coordinates = TODO
delays = np.zeros(n_obs_points, *grid_size)

for i, r in enumerate(r_obs):
    delays[i] = (r - grid_coordinates)/speed_of_light

# Observation time range
dt = (t.max() - t.min()) / (len(t) - 1)
t_obs = np.arange(t.min() + delays.min(),
                  t.max() + delays.max(), dt)
n_obs_times = len(t_obs)

E_obs = np.zeros(n_obs_times, n_obs_points, 3)
factors = 1 / (4 * np.pi * epsilon_0 * speed_of_light * norm(r_obs, axis=1))

for k, t in enumerate(t_obs):
    for i, r in enumerate(r_obs):
        t_source = t - delays[i]

        # Approximate unit vector from observation to source (since source
        # region is much smaller than norm(r))
        r_hat = -r / norm(r)

        # Interpolate at every grid point at the given t_source
        rho_term = r_hat (TODO outer product) drho_dt(t_source)
        J_term = dJ_dt(t_source) / speed_of_light

        E_obs[k, i] = E_obs[k, i] + dV * factors[i] * (rho_term - J_term)
