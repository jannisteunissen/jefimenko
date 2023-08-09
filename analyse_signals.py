#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse
from scipy.signal import savgol_filter
from scipy.signal import welch
from os.path import splitext
# Below modules are needed to perform wavelet analysis
import scaleogram as scg 
import pywt

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Some sort of signal analysis of the emitted field')
p.add_argument('data_files', type=str, nargs="+",
               help='''CSV files of different points to visualize''')
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
for i, f in enumerate(args.data_files):
    fig, ax = plt.subplots(1,4, figsize=(20,5))
    # Get r_obs from file name
    tmp = splitext(f)[0]
    # Get coordinate string
    tmp = tmp.split('_observer_')[-1]
    r = ' '.join(tmp.split('_'))

    plt.suptitle(f"Signal analysis for observer at r=[{r}]")
    # Reading the File
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    t_obs = data[:, 0]
    t_src = data[:, 1]
    E_obs_rho = data[:, 2:5]
    E_obs_J = data[:, 5:]
    
    # Visualizing the total signal
    y_scaling = 1.0
    E_obs = E_obs_rho + E_obs_J
    ax[0].plot(t_src, y_scaling*custom_filter(E_obs[:,0]), label="E_x")
    ax[0].plot(t_src, y_scaling*custom_filter(E_obs[:,1]), label="E_y")
    ax[0].plot(t_src, y_scaling*custom_filter(E_obs[:,2]), label="E_z")
    ax[0].set_title(f"Radiated field * {y_scaling}")
    ax[0].set_xlabel("Time (sec)")
    #ax[0].set_yscale("log")
    ax[0].legend()

    # Visualizing the spectrograms of the signal
    # in each direction
    # TODO: Need to finetune the parameters used to plot the spectrogram
    # dt_inv = 1/(t_src[1] - t_src[0])
    # _, _, _, p1 = ax[1].specgram(custom_filter(E_obs[:,0]), 
    #                     NFFT=32, Fs=dt_inv, noverlap=31, cmap="jet_r")
    # _, _, _, p2 = ax[2].specgram(custom_filter(E_obs[:,1]), 
    #                     NFFT=32, Fs=dt_inv, noverlap=31, cmap="jet_r")
    # ax[2].set_title("Spectrogram of E_y")
    # _, _, _, p3 = ax[3].specgram(custom_filter(E_obs[:,2]), 
    #                              NFFT=32, Fs=dt_inv, noverlap=31, cmap="jet_r")
    # ax[1].set_title("Spectrogram of E_x")
    # ax[2].set_title("Spectrogram of E_y")
    # ax[3].set_title("Spectrogram of E_z")
    # fig.colorbar(p1, ax=ax[1])
    # fig.colorbar(p2, ax=ax[2])
    # fig.colorbar(p3, ax=ax[3])

    # Visualizing the scaleograms of the signal
    # in each direction
    # TODO: Need to finetune the parameters used to plot the scaleogram
    scg.set_default_wavelet('cmor1-1.5')
    scales = scg.periods2scales( np.arange(1, 40) )
    scg.cws(t_src, signal=custom_filter(E_obs[:,0]), scales=scales,
            yaxis="frequency", ax = ax[1])
    scg.cws(t_src, signal=custom_filter(E_obs[:,1]), scales=scales,
            yaxis="frequency", ax = ax[2])
    scg.cws(t_src, signal=custom_filter(E_obs[:,2]), scales=scales,
            yaxis="frequency", ax = ax[3])
    ax[1].set_title("Scaleogram of E_x")
    ax[2].set_title("Scaleogram of E_y")
    ax[3].set_title("Scaleogram of E_z")

plt.show()
