#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from scipy.signal import savgol_filter
from os.path import splitext
# Below modules are needed to perform wavelet analysis
import scaleogram as scg
import pywt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["axes.labelsize"] = 20

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Some sort of signal analysis of the emitted field')
p.add_argument('data_files', type=str, nargs="+",
               help='''CSV files of different points to visualize''')
p.add_argument('-save_figname', type=str, default="out_cwt.pdf",
               help='Save the figures generated as a pdf')
p.add_argument('-ignore_starting_points', type=int, default=0,
               help='''Number of starting points to ignore while plotting''')
p.add_argument('-ignore_end_points', type=int, default=0,
               help='''Number of ending points to ignore while plotting''')
p.add_argument('-x_y_scale', type=float, nargs=2, default=[1e9, 1e6],
               help='x and y values are multiplied with these values')
p.add_argument('-clims', type=float, nargs=2, default=[None, None],
               help='''Limits of the scaleogram colorbar''')
p.add_argument('-x_lims', type=float, nargs=2, default=[None, None],
               help='x limts used for plotting')
p.add_argument('-y_lims', type=float, nargs=2, default=[0.02, 1],
               help='y limts used for plotting')
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



data_files = args.data_files
N = len(data_files)
in_to_mm = 1/25.4
for i, f in enumerate(data_files):
    fig, ax = plt.subplots(1, 4, figsize=(4*115*in_to_mm, 2*95*in_to_mm),
                           constrained_layout=True)
    # Get r_obs from file name
    tmp = splitext(f)[0]
    # Get coordinate string
    tmp = tmp.split('_observer_')[-1]
    r = ' '.join(tmp.split('_'))

    # Reading the File
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    ignored_pts = args.ignore_starting_points
    data = data[ignored_pts:, :]
    if args.ignore_end_points > 0:
        data = data[:-args.ignore_end_points, :]

    x_scale, y_scale = args.x_y_scale
    data[:, 0:2] = x_scale*data[:, 0:2]
    t_obs = data[:, 0]
    t_src = data[:, 1]
    E_obs_static = data[:, 2:5]
    E_obs_rho = data[:, 5:8]
    E_obs_J = data[:, 8:]

    # Visualizing the total signal
    E_obs = E_obs_rho + E_obs_J + E_obs_static
    ax[0].plot(t_src, y_scale*custom_filter(E_obs[:, 0]),
               label="$E_x$")
    ax[0].plot(t_src, y_scale*custom_filter(E_obs[:, 1]),
               label="$E_y$")
    ax[0].plot(t_src, y_scale*custom_filter(E_obs[:, 2]),
               label="$E_z$")
    ax[0].set_xlabel(f"Time (ns)")
    ax[0].set_ylabel("$\mathrm{E}_\mathrm{rad}$ ($\mu$V/m)")
    ax[0].legend()

    dt_inv = 1/(t_src[1] - t_src[0])
    # Visualizing the scaleograms of the signal
    # in each direction

    waveletname = 'cmor1.5-1.0'
    scg.set_default_wavelet(waveletname)
    # Resolving frequencies from 0.1 Hz to 100Hz
    freqs = pywt.scale2frequency(waveletname, np.logspace(-1, 2, num=400))
    scales = scg.periods2scales(1./freqs)
    # Scpectrum to use- 'amp','real', 'imag', 'power', 'other*'
    spec = 'amp'
    direc = ["x", "y", "z"]
    if not all(args.clims):
        cbarlims = None
    else:
        cbarlims = (args.clims[0], args.clims[1])
    for i in range(1, 4):
        if not cbarlims:
            if i == 3:
                scg.cws(t_src, signal=custom_filter(E_obs[:, i-1]), scales=scales,
                        yaxis="frequency", spectrum=spec, ax = ax[i], coi=False, 
                        cscale="log", yscale="log",
                        title=f"Scaleogram of $E_{direc[i-1]}$")
            else:
                scg.cws(t_src, signal=custom_filter(E_obs[:, i-1]), scales=scales,
                        yaxis="frequency", spectrum=spec, ax = ax[i], coi=False, 
                        cscale="log", yscale="log", cbar=None,
                        title=f"Scaleogram of $E_{direc[i-1]}$")
        # yes colorbar
        else:
            if i == 3:
                scg.cws(t_src, signal=custom_filter(E_obs[:, i-1]), scales=scales,
                        yaxis="frequency", spectrum=spec, ax=ax[i], coi=False,
                        cscale="log", yscale="log", clim=cbarlims, cbar='vertical',
                        title=f"Scaleogram of $E_{direc[i-1]}$")
            else:
                scg.cws(t_src, signal=custom_filter(E_obs[:, i-1]), scales=scales,
                        yaxis="frequency", spectrum=spec, ax=ax[i], coi=False,
                        cscale="log", yscale="log", clim=cbarlims, cbar=None,
                        title=f"Scaleogram of $E_{direc[i-1]}$")

    ax[1].get_shared_y_axes().join(ax[1], *ax[2:])
    ax[2].axes.get_yaxis().set_visible(False)
    ax[3].axes.get_yaxis().set_visible(False)
    ax[1].set_xlabel("Time (ns)")
    ax[2].set_xlabel("Time (ns)")
    ax[3].set_xlabel("Time (ns)")
    ax[1].set_ylabel("Freq (Ghz)")
    ax[2].set_ylabel(None)
    ax[3].set_ylabel(None)

    # Saving the figures
    if all(args.y_lims):
        ax[1].set_ylim(args.y_lims)
    if all(args.x_lims):
        plt.setp(ax, xlim=(args.x_lims[0], args.x_lims[1]))
    fig.savefig(args.save_figname, transparent=True)
    print("Saved figure as "+args.save_figname)
plt.show()
