#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm
import argparse
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib import colors, animation

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Visualize the supplied npz file over time')
p.add_argument('npz', type=str,
               help='''Numpy file with 4D charge density scalar rho(t,x,y,z),
               5D current density vector J(t,dim,x,y,z), time array t and
               grid coordinate arrays x, y, z''')
args = p.parse_args()

file_prefix = splitext(args.npz)[0]
npz_file = np.load(args.npz)
t = npz_file['t']
x, y, z = npz_file['x'], npz_file['y'], npz_file['z']
Nx = x.shape[0]

# Assume constant dx
dx = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
dV = np.product(dx)
grid_coordinates = np.meshgrid(x, y, z, indexing='ij', sparse=True)

N_t = len(t) - 1

# Load all data
rho_samples = npz_file['rho']
J_samples = npz_file['J']
grid_size = rho_samples.shape[1:]


fig, ax = plt.subplots(1,2, figsize=(16,8))
domain_size = [x[0], x[-1], z[0], z[-1]]
rho_lims = [np.min(rho_samples), np.max(rho_samples)]
ax[0].set_title("Charge density")
ax[1].set_title("Current density magnitude")


def update_plots(i):
    J_slice = norm(J_samples[i, :, :, Nx//2, :], axis=0)
    J_lims = [np.min(J_slice), np.max(J_slice)]
    plt.suptitle(f"Time = {t[i]}sec")
    p1 = ax[0].imshow(rho_samples[i][:, Nx//2, :], origin="lower",
                      vmin=rho_lims[0], vmax=rho_lims[1], 
                      extent=domain_size,
                      cmap="plasma")
    p2 = ax[1].imshow(J_slice+1, origin="lower", 
                      norm=colors.LogNorm(vmin=J_lims[0]+1, vmax=J_lims[1]),
                      extent=domain_size,
                      cmap="jet") 


# Plotting the first timestep
i=0
J_slice = norm(J_samples[i, :, :, Nx//2, :], axis=0)
J_lims = [np.min(J_slice), np.max(J_slice)]
p1 = ax[0].imshow(rho_samples[i][:, Nx//2, :], origin="lower",
                  vmin=rho_lims[0], vmax=rho_lims[1], 
                  extent=domain_size,
                  cmap="plasma")
p2 = ax[1].imshow(J_slice+1, origin="lower", 
                  norm=colors.LogNorm(vmin=J_lims[0]+1, vmax=J_lims[1]),
                  extent=domain_size,
                  cmap="jet") 
fig.colorbar(p1, ax=ax[0])
fig.colorbar(p2, ax=ax[1])
plt.suptitle(f"Time = {t[i]}sec")

anim = animation.FuncAnimation(fig, update_plots, frames = N_t)
#anim.save(filename=file_prefix+".mp4", fps=30, extra_args=['-vcodec', 'libx264'])



plt.show()
