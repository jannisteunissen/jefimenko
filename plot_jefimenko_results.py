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
               help='''CSV files of different points to visualize''')
p.add_argument('-filter_width', type=int, default=0,
               help='''Odd number, if > 0, apply savgol_filter''')
p.add_argument('-filter_order', type=int, default=1,
               help='''Order of the filter''')
p.add_argument('-save_figs', type=str, default="out_Efldspectrum.png",
                    help='Save the figures generated as a png')
args = p.parse_args()


def custom_filter(x):
    if args.filter_width > 0:
        return savgol_filter(x, args.filter_width, args.filter_order)
    else:
        return x


N = len(args.data_files)
fig, ax = plt.subplots(N, 3, sharex='row', sharey=True, figsize=(8, N*3))
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
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    t_obs = data[:, 0]
    t_src = data[:, 1]
    E_obs_rho = data[:, 2:5]
    E_obs_J = data[:, 5:]
    ax[i, 0].plot(t_src, custom_filter(E_obs_rho[:, 0]), label='Ex')
    ax[i, 0].plot(t_src, custom_filter(E_obs_rho[:, 1]), label='Ey')
    ax[i, 0].plot(t_src, custom_filter(E_obs_rho[:, 2]), label='Ez')
    tmp = savgol_filter(E_obs_rho[:, 2], 13, 1)
    ax[i, 0].legend()
    ax[i, 0].set_ylabel('E (V/m)')

    ax[i, 1].plot(t_src, custom_filter(E_obs_J[:, 0]), label='Ex')
    ax[i, 1].plot(t_src, custom_filter(E_obs_J[:, 1]), label='Ey')
    ax[i, 1].plot(t_src, custom_filter(E_obs_J[:, 2]), label='Ez')

    ax[i, 2].plot(t_src, custom_filter(E_obs_rho[:, 0] + E_obs_J[:, 0]),
                  label='Ex')
    ax[i, 2].plot(t_src, custom_filter(E_obs_rho[:, 1] + E_obs_J[:, 1]),
                  label='Ey')
    ax[i, 2].plot(t_src, custom_filter(E_obs_rho[:, 2] + E_obs_J[:, 2]),
                  label='Ez')

    ax[i, 2].text(1.1, 0.5, r'$r_\mathrm{obs}$ = ' + r,
                  rotation='vertical',
                  horizontalalignment='center',
                  verticalalignment='center', transform=ax[i, 2].transAxes)

    # Power spectrum
    freq, Pxx_den = welch(custom_filter(E_obs_rho[:, 2]),
                          1/(t_src[1] - t_src[0]))
    ax2[i].semilogy(freq * 1e-6, Pxx_den)
    ax2[i].set_xlabel('Frequency (MHz)')
    ax2[i].set_ylabel('PSD [V**2/Hz]')
    ax2[i].set_title(f'Observer {i} at {r}')

# Set labels only for certain plots
ax[0, 0].set_title(r'$\partial_t \rho$ term')
ax[0, 1].set_title(r'$\partial_t \vec{J}$ term')
ax[0, 2].set_title(f'total')

ax[-1, 0].set_xlabel('t (s)')
ax[-1, 1].set_xlabel('t (s)')
ax[-1, 2].set_xlabel('t (s)')

#Saving the figures
if args.save_figs.startswith("out"):
    fig.savefig("out_Efld.png", dpi=100)
    fig2.savefig("out_spectrum.png", dpi=100)
else:
    fig.savefig(args.save_figs+"_Efld.png", dpi=100)
    fig2.savefig(args.save_figs+"_spectrum.png", dpi=100)
    

plt.tight_layout()
plt.show()
