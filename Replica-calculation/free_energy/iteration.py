import os
import sys
wd = os.path.dirname(os.path.abspath(__file__))+'/../'
sys.path.append(wd)
from utils import *

# parameter_model = (g, eta)
# parameter_numeric = (number_Gaussian_sample, bound_x, numeric_precision)
# parameter_iteration = (max_iteration, damping_coefficient,
#                        data_path, log_interval, convergence_precision, min_iteration)
# initialization = torch.tensor([q, delta_Q, delta_q_hat, Q_hat])


class iteration:
    def __init__(self, parameter_model, parameter_numeric, parameter_iteration, initialization):
        self.parameter_model = parameter_model
        self.parameter_numeric = parameter_numeric
        self.parameter_iteration = parameter_iteration
        self.value = initialization
        self.value_history = [self.value.clone()]

    def iterate(self):
        print('start', time.asctime(time.localtime(time.time())))

        for i in range(self.parameter_iteration[0]):
            value, self.x_star, self.z2 = self.map()
            self.update(value)
            error_occur = self.error()
            self.save()
            self.log(i)
            if (i+1) > self.parameter_iteration[5] and not error_occur and self.is_converged():
                break

        print('end', time.asctime(time.localtime(time.time())))
        if (i+1) < self.parameter_iteration[0]:
            print('convergence reached in iteration', i)
        else:
            print('maximum number of iterations reached')

    def map(self):
        z1 = torch.randn(self.parameter_numeric[0])
        z2 = torch.randn(self.parameter_numeric[0])
        sigma = torch.sqrt(1+self.parameter_model[0]**2*self.value[1])

        def H(x):
            return (self.parameter_model[1]+1/(2 * sigma**2))*x**2\
                - self.value[2]*phi(x)**2\
                - z1 * torch.sqrt(self.value[3])*phi(x)\
                - z2*self.parameter_model[0] * torch.sqrt(self.value[0])/sigma**2*x
        optimize_result_left = golden_section_search(
            H, torch.ones(self.parameter_numeric[0])*self.parameter_numeric[1][0], torch.zeros(self.parameter_numeric[0]), self.parameter_numeric[2])
        optimize_result_right = golden_section_search(
            H, torch.zeros(self.parameter_numeric[0]), torch.ones(self.parameter_numeric[0])*self.parameter_numeric[1][1], self.parameter_numeric[2])
        x_star = optimize_result_left[0].clone()
        left_greater_right = optimize_result_left[1] > optimize_result_right[1]
        x_star[left_greater_right] = optimize_result_right[0][left_greater_right]

        q = torch.mean(phi(x_star)**2)
        delta_Q = 1/torch.sqrt(self.value[3])*torch.mean(z1*phi(x_star))
        delta_q_hat = -self.parameter_model[0]**2/(2*sigma**2)\
            + self.parameter_model[0] / (2*sigma**2 *
                                         torch.sqrt(self.value[0]))*torch.mean(z2*x_star)
        Q_hat = self.parameter_model[0]**4*self.value[0]/sigma**4\
            + self.parameter_model[0]**2/sigma**4*torch.mean(x_star**2)\
            - 2*self.parameter_model[0]**3 * \
            torch.sqrt(self.value[0])/sigma**4*torch.mean(z2*x_star)

        return (torch.tensor([q, delta_Q, delta_q_hat, Q_hat]), x_star, z2)

    def update(self, value):
        self.value = self.parameter_iteration[1] * \
            self.value + (1-self.parameter_iteration[1]) * value

    def error(self):
        error_occur = False
        if self.value[0] < 0:
            print('q<0')
            self.value[0] = self.parameter_numeric[2]
            error_occur = True
        if self.value[1] < 0:
            print('delta_Q<0')
            self.value[1] = self.parameter_numeric[2]
            error_occur = True
        if self.value[3] < 0:
            print('Q_hat<0')
            self.value[3] = self.parameter_numeric[2]
            error_occur = True
        return error_occur

    def save(self):
        self.value_history.append(self.value.clone())
        torch.save((self.value_history, self.x_star, self.z2),
                   self.parameter_iteration[2]+'iteration.pt')

    def log(self, i):
        if (i + 1) % self.parameter_iteration[3] == 0:
            print(i, time.asctime(time.localtime(time.time())))
            print('q', self.value[0].item())

    def is_converged(self):
        return torch.abs((self.value_history[-1][0]-self.value_history[-2][0])) < self.parameter_iteration[4]
