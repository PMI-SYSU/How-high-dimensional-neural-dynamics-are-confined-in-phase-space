# version: main.minor changes

from iteration import *
import os
import sys
wd = os.path.dirname(os.path.abspath(__file__))+'/../'
sys.path.append(wd)
from utils import *
path = wd+'../Data/RNN-fixed-point-local-entropy/12/'

g_list=(.9,1.1,1.2)
eta = 0.
p_hat_list = (-1e6,1e6)
beta_tilde = 1e6

number_Gaussian_sample = (int(1e5), int(1e4))
numeric_precision = 1e-6

max_iteration = int(1e2)
damping_coefficient = .5
log_interval = int(1e1)

repetition = 1

for i, g, p_hat in itertools.product(range(repetition), g_list, p_hat_list, ):    
    print('g', g, 'p_hat', p_hat, 'beta_tilde', beta_tilde, i)

    result_free_energy = torch.load(
        path+'g_'+str(g)+'_'+str(i)+'_iteration.pt')
    result_free_energy = (
        result_free_energy[0][-1][0], result_free_energy[1], result_free_energy[2])
    data_path = path+'g_' + \
        str(g)+'_beta_tilde_'+str(beta_tilde)+'_p_hat_'+str(p_hat)+'_'+str(i)+'_'
    o = torch.randn(1).item()
    q_tilde_hat = torch.randn(1).item()
    Q_tilde_hat = torch.rand(1).item()
    delta_o_hat = torch.randn(1).item()
    Q_tilde = o**2/result_free_energy[0] + numeric_precision
    q_tilde = Q_tilde + numeric_precision

    parameter_model = (g, eta, p_hat, beta_tilde)
    parameter_numeric = (result_free_energy,
                         number_Gaussian_sample, numeric_precision)
    parameter_iteration = (
        max_iteration, damping_coefficient, data_path, log_interval)
    initialization = torch.tensor(
        [q_tilde, Q_tilde, o, q_tilde_hat, Q_tilde_hat, delta_o_hat])

    SDE = iteration(parameter_model, parameter_numeric,
                    parameter_iteration, initialization)
    SDE.iterate()
