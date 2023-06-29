#!/usr/bin/env python3

import numpy as np
import argparse
from scipy.constants import epsilon_0

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''Combine a series of npz files into a combined file for
    the Jefimenko script''')
parser.add_argument('npz', type=str, nargs='+',
                    help='Npz files containing rhs and Je terms')
parser.add_argument('-output', type=str, default='combined_data.npz',
                    help='Output filename')
args = parser.parse_args()

args.npz = sorted(args.npz)
rhs_files = [f for f in args.npz if '_rhs.npz' in f]
Je_1_files = [f for f in args.npz if '_Je_1.npz' in f]
Je_2_files = [f for f in args.npz if '_Je_2.npz' in f]
Je_3_files = [f for f in args.npz if '_Je_3.npz' in f]

N = len(rhs_files)
tmp = np.load(rhs_files[0])
data_shape = tmp['uniform_data'].shape
x, y, z = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
t = np.zeros(N)

rhs = np.zeros([N, *data_shape])
J = np.zeros([N, 3, *data_shape])

for i in range(N):
    rhs_data = np.load(rhs_files[i])
    t[i] = rhs_data['time']
    rhs[i] = rhs_data['uniform_data'] * -epsilon_0
    J[i, 0] = np.load(Je_1_files[i])['uniform_data']
    J[i, 1] = np.load(Je_2_files[i])['uniform_data']
    J[i, 2] = np.load(Je_3_files[i])['uniform_data']

np.savez(args.output, rho=rhs, J=J, x=x, y=y, z=z, t=t)
print(f'Written {args.output}')
