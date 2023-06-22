#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse
from scipy.signal import savgol_filter
from scipy.signal import welch
from os.path import splitext

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('data_files', type=str, nargs="+",
               help='''Files of different points to visualize''')
p.add_argument('-filter_width', type=int, default=0,
               help='''Odd number, if > 0, apply savgol_filter''')
p.add_argument('-filter_order', type=int, default=1,
               help='''Order of the filter''')
args = p.parse_args()


def custom_filter(x):
    if args.filter_width > 0:
        return savgol_filter(x, args.filter_width, args.filter_order)
    else:
        return x


N = len(args.data_files)
fig, ax = plt.subplots(N, 3, sharex='row', sharey=True)
fig2, ax2 = plt.subplots(N, sharex=True, sharey=True)

if N == 1:
    ax = ax[None, :]
    ax2 = [ax2]

for i, f in enumerate(args.data_files):
    # Get r_obs from file name
    tmp = splitext(f)[0]
    # Get coordinate string
    tmp = tmp.split('_observer_')[-1]
    r = ' '.join(tmp.split('_'))

    # Reading the File
    data = np.loadtxt(f)
    t_obs = data[:, 0]
    E_obs_rho = data[:, 1:4]
    E_obs_J = data[:, 4:]
    ax[i, 0].plot(t_obs, custom_filter(E_obs_rho[:, 0]), label='Ex')
    ax[i, 0].plot(t_obs, custom_filter(E_obs_rho[:, 1]), label='Ey')
    ax[i, 0].plot(t_obs, custom_filter(E_obs_rho[:, 2]), label='Ez')
    tmp = savgol_filter(E_obs_rho[:, 2], 13, 1)
    ax[i, 0].legend()
    ax[i, 0].set_title('rho')
    ax[i, 0].set_xlabel('t (s)')
    ax[i, 0].set_ylabel('E (V/m)')

    ax[i, 1].plot(t_obs, custom_filter(E_obs_J[:, 0]), label='Ex')
    ax[i, 1].plot(t_obs, custom_filter(E_obs_J[:, 1]), label='Ey')
    ax[i, 1].plot(t_obs, custom_filter(E_obs_J[:, 2]), label='Ez')
    ax[i, 1].legend()
    ax[i, 1].set_title('J')
    ax[i, 1].set_xlabel('t (s)')

    ax[i, 2].plot(t_obs, custom_filter(E_obs_rho[:, 0] + E_obs_J[:, 0]),
                  label='Ex')
    ax[i, 2].plot(t_obs, custom_filter(E_obs_rho[:, 1] + E_obs_J[:, 1]),
                  label='Ey')
    ax[i, 2].plot(t_obs, custom_filter(E_obs_rho[:, 2] + E_obs_J[:, 2]),
                  label='Ez')
    ax[i, 2].legend()
    ax[i, 2].set_title(f'total')
    ax[i, 2].set_xlabel('t (s)')

    # Power spectrum
    freq, Pxx_den = welch(custom_filter(E_obs_rho[:, 2]),
                          1/(t_obs[1] - t_obs[0]))
    ax2[i].semilogy(freq * 1e-6, Pxx_den)
    ax2[i].set_xlabel('Frequency (MHz)')
    ax2[i].set_ylabel('PSD [V**2/Hz]')
    ax2[i].set_title(f'Observer {i} at {r}')

plt.tight_layout()
plt.show()
