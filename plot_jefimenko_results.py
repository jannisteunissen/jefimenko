#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('-data_files', type=str, required=True, nargs="+",
               help='''Files of different points to visualize''')
p.add_argument('-observation_points', type=float, nargs='+', required=True,
               help='N*3 coordinates of outside observation points')
args = p.parse_args()


r_obs = np.reshape(args.observation_points, [-1, 3])
n_obs_points = len(r_obs)

fig, ax = plt.subplots(n_obs_points, 3, sharex='row', sharey=True)
if ax.ndim == 1:
    ax = ax[None, :]

for i, r in enumerate(r_obs):
    # Reading the File
    data = np.loadtxt(args.data_files[i])
    t_obs = data[:, 0]
    E_obs_rho = data[:, 1:4]
    E_obs_J = data[:, 4:]
    ax[i, 0].plot(t_obs, E_obs_rho[:, 0], label='Ex')
    ax[i, 0].plot(t_obs, E_obs_rho[:, 1], label='Ey')
    ax[i, 0].plot(t_obs, E_obs_rho[:, 2], label='Ez')
    ax[i, 0].legend()
    ax[i, 0].set_title(f'Observer {i+1} at {r} (rho)')

    ax[i, 1].plot(t_obs, E_obs_J[:, 0], label='Ex')
    ax[i, 1].plot(t_obs, E_obs_J[:, 1], label='Ey')
    ax[i, 1].plot(t_obs, E_obs_J[:, 2], label='Ez')
    ax[i, 1].legend()
    ax[i, 1].set_title(f'Observer {i+1} at {r} (J)')

    ax[i, 2].plot(t_obs, E_obs_rho[:, 0] + E_obs_J[:, 0], label='Ex')
    ax[i, 2].plot(t_obs, E_obs_rho[:, 1] + E_obs_J[:, 1], label='Ey')
    ax[i, 2].plot(t_obs, E_obs_rho[:, 2] + E_obs_J[:, 2], label='Ez')
    ax[i, 2].plot(t_obs, norm(E_obs_rho + E_obs_J, axis=1),
                  '--', label='||E||')
    ax[i, 2].legend()
    ax[i, 2].set_title(f'Observer {i+1} at {r} (total)')

plt.show()
