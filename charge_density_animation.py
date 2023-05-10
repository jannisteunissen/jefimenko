from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical solution to Jefimenko equations')
p.add_argument('-npz', type=str, required=True,
               help='''Numpy file with 3D charge density scalar rho(t,x,y),
               4D current density vector J(t,dim,x,y), time array t and
               grid coordinate arrays x, y''')
args = p.parse_args()



def generate(X, Y, t):
    mu = 0.2*np.sin(2*np.pi*2*t)
    sigma = 0.2
    R = np.exp(- (X**2 + (Y-mu)**2)/(2*sigma**2))
    denom = 1.0 # 2*np.pi*sigma**2
    return R/denom


fig = plt.figure()
ax = axes3d.Axes3D(fig)

npz_file = np.load(args.npz)
t = npz_file["t"]
x, y = npz_file["x"], npz_file["y"]
X, Y = np.meshgrid(x, y)
Z = npz_file['rho']
wframe = ax.plot_wireframe(X, Y, Z[0], rstride=2, cstride=2)
ax.set_zlim(-4, 4)


def update(i, ax, fig):
    ax.cla()
    t = i/50.0
    # Z = generate(X, Y, t)
    wframe = ax.plot_wireframe(X, Y, Z[i], rstride=2, cstride=2)
    ax.set_zlim(-4, 4)
    return wframe,


ani = animation.FuncAnimation(fig, update, frames=range(len(t)), fargs=(ax, fig), interval=1)
plt.show()
