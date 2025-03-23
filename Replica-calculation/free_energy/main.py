# version: main.basic

from iteration import *
import os
import sys
wd = os.path.dirname(os.path.abspath(__file__))+'/../'
sys.path.append(wd)
from utils import *
path = wd+'../Data/RNN-fixed-point-local-entropy/12/'

g_list=(.9,1.1,1.2)
eta = 0.

number_Gaussian_sample = int(1e6)
bound_x = (-1e1, 1e1)
numeric_precision = 1e-6

max_iteration = int(1e3)
damping_coefficient = .5
log_interval = int(1e1)
convergence_precision = 1e-6
min_iteration = int(1e2)

repetition = 1

for i, g in itertools.product(range(repetition), g_list):
    print('g', g, i)

    data_path = path + 'g_' + str(g)+'_'+str(i)+'_'
    q = torch.rand(1).item()
    delta_Q = torch.rand(1).item()
    delta_q_hat = torch.randn(1).item()
    Q_hat = torch.rand(1).item()

    parameter_model = (g, eta)
    parameter_numeric = (number_Gaussian_sample, bound_x, numeric_precision)
    parameter_iteration = (max_iteration, damping_coefficient,
                           data_path, log_interval, convergence_precision, min_iteration)
    initialization = torch.tensor([q, delta_Q, delta_q_hat, Q_hat])

    SDE = iteration(parameter_model, parameter_numeric,
                    parameter_iteration, initialization)
    SDE.iterate()
