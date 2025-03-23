import numpy as np
import h5py
import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir)

g = 3.0
N = 100
Delta_t = 0.1
x_len = 10
with h5py.File(dir+'/finder.h5', 'r') as f:
    x = f['x'][:x_len]
J = np.load(dir+'/J.npy')
epsilon = 1e-6
T = 1e-6
log_interval = 100
