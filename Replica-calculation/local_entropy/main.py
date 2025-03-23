from iteration import *
import os
import sys
wd = os.path.dirname(os.path.abspath(__file__))+'/../'
sys.path.append(wd)
from utils import *
path = wd+'../Data/RNN-fixed-point-local-entropy/data/'

g = 1.1
eta = 0.
beta_tilde = 1e6

number_Gaussian_sample = (int(1e5), int(1e4))
numeric_precision = 1e-6
initialization_secant = (-1., 1.)

max_iteration = int(1e2)
damping_coefficient = .5
log_interval = int(1e2)
convergence_precision = (1e-3, 1e-2)
mean_interval = int(1e2)

p_hat_list = (-1e6, 1e6)
i = 0
n = int(1e2)


def d_end(p_hat):
    return torch.load(path+'g_' + str(g)+'_beta_tilde_'+str(beta_tilde)+'_p_hat_'+str(p_hat)+'_'+str(i)+'_iteration.pt', map_location='cpu')[1][-1]


d_list = np.linspace(d_end(p_hat_list[0]), d_end(p_hat_list[1]), n+1)[1:-1]

for d in d_list:
    print('g', g, 'd', d)

    result_free_energy = torch.load(
        path + 'g_'+str(g)+'_'+str(i)+'_iteration.pt')
    result_free_energy = (
        result_free_energy[0][-1][0], result_free_energy[1], result_free_energy[2])
    data_path = path + 'local_entropy/g_' + str(g)+'_beta_tilde_'+str(beta_tilde)+'_d_'+str(d)+'_'

    parameter_model = (g, eta, d, beta_tilde)
    parameter_numeric = (result_free_energy, number_Gaussian_sample,
                         numeric_precision, initialization_secant)
    parameter_iteration = (max_iteration, damping_coefficient,
                           data_path, log_interval, convergence_precision, mean_interval)
    initialization = torch.load(path + 'g_' + str(g)+'_beta_tilde_'+str(beta_tilde)+'_d_'+str(d)+'_iteration.pt')[0][-1]

    SDE = iteration(parameter_model, parameter_numeric,
                    parameter_iteration, initialization)
    SDE.iterate()
