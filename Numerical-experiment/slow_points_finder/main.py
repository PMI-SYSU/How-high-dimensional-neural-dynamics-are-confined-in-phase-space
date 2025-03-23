import os
import sys
sys.path.append(
    '/Users/shihsherwang/Local/Github/RNN-fixed-point-distribution/slow_points_finder')
dir = os.path.dirname(os.path.abspath(__file__))
from finder import *

g = 3.
N = 100
epsilon = 1e-6
step = int(1e6)
std = 100.
tolerance = 1e-3
proportion = 1e-2
data_path = dir+'/'
log_interval = 1000

slow_point_finder = finder(g, N, epsilon)
slow_point_finder.main(step, std, tolerance, proportion,
                       data_path, log_interval)
