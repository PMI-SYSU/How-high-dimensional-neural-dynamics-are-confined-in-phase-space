import os
import sys
wd = os.path.dirname(os.path.abspath(__file__))+'/../'
sys.path.append(wd)
from utils import *

# parameter_model = (g, eta, d, beta_tilde)
# parameter_numeric = (result_free_energy, number_Gaussian_sample, numeric_precision, initialization_secant)
# parameter_iteration = (max_iteration, damping_coefficient, data_path, log_interval, convergence_precision, mean_interval)
# initialization = torch.tensor([q_tilde, Q_tilde, o, q_tilde_hat, Q_tilde_hat, delta_o_hat])


class iteration:
    def __init__(self, parameter_model, parameter_numeric, parameter_iteration, initialization):
        self.parameter_model = parameter_model
        self.parameter_numeric = parameter_numeric
        self.parameter_iteration = parameter_iteration
        self.value = initialization
        self.value_history = [self.value.clone()]
        self.local_entropy_history = []
        self.derivative_history = []

    def iterate(self):
        print('start:', time.asctime(time.localtime(time.time())))

        for i in range(self.parameter_iteration[0]):
            value, self.local_entropy, self.derivative = self.map()
            self.update(value)
            error_occur = self.error()
            self.save()
            self.log(i)
            if (i+1) % self.parameter_iteration[5] == 0:
                j = (i+1)/self.parameter_iteration[5]
                if j > 1:
                    if self.is_converged(j):
                        break

        print('end', time.asctime(time.localtime(time.time())))
        if (i + 1) < self.parameter_iteration[0]:
            print('convergence reached in iteration', i)
        else:
            print('maximum number of iterations reached')

    def map(self):
        q = self.parameter_numeric[0][0]
        x_star = self.parameter_numeric[0][1][:self.parameter_numeric[1][0]]
        z2 = self.parameter_numeric[0][2][:self.parameter_numeric[1][0]]
        z3 = torch.randn(self.parameter_numeric[1][0])
        z4 = torch.randn(self.parameter_numeric[1][0])
        y_prime = torch.randn(self.parameter_numeric[1])
        sigma_tilde = torch.sqrt(
            1+self.parameter_model[0]**2*self.parameter_model[3] * (self.value[0]-self.value[1]))
        k_tilde = self.parameter_model[0] * \
            self.parameter_model[3]/sigma_tilde**2
        sigma_prime = 1 / \
            torch.sqrt(
                2*self.parameter_model[3]*self.parameter_model[1]+k_tilde/self.parameter_model[0])
        y = sigma_prime*y_prime

        H_tilde_prime_p_hat_missing = (self.value[3]-1/2*self.value[4])*phi(y)**2\
            + z3[:, None]*torch.sqrt(self.value[4])*phi(y)\
            + self.value[5]*phi(x_star[:, None])*phi(y)\
            + k_tilde*(z2[:, None]*self.value[2]/torch.sqrt(q)
                       + z4[:, None]*torch.sqrt(self.value[1]-self.value[2]**2/q))*y

        def d_2(p_hat):
            H_tilde_prime = H_tilde_prime_p_hat_missing + \
                p_hat*(y-x_star[:, None])**2
            exp_H_tilde_prime_alternative = torch.exp(
                H_tilde_prime-torch.max(H_tilde_prime, dim=1)[0][:, None])
            return torch.mean(torch.mean(exp_H_tilde_prime_alternative*(y-x_star[:, None])**2, dim=1)/torch.mean(exp_H_tilde_prime_alternative, dim=1))
        root_results = optimize.root_scalar(lambda p_hat: np.array(
            (d_2(p_hat)-self.parameter_model[2]**2).cpu()), method='secant', x0=self.parameter_numeric[3][0], x1=self.parameter_numeric[3][1])
        p_hat = root_results.root.astype(np.float32)
        H_tilde_prime = H_tilde_prime_p_hat_missing + \
            p_hat*(y-x_star[:, None])**2
        exp_H_tilde_prime_alternative = torch.exp(
            H_tilde_prime-torch.max(H_tilde_prime, dim=1)[0][:, None])

        def curly_average(f):
            return torch.mean(exp_H_tilde_prime_alternative*f, dim=1)/torch.mean(exp_H_tilde_prime_alternative, dim=1)
        q_tilde = torch.mean(curly_average(phi(y)**2))
        Q_tilde = torch.mean(curly_average(phi(y))**2)
        o = torch.mean(curly_average(phi(x_star[:, None])*phi(y)))
        q_tilde_hat = -1/2*self.parameter_model[0]*k_tilde\
            + 1/2*self.parameter_model[0]**2*k_tilde**2*self.value[1]\
            + 1/2*k_tilde**2 * (1-2*self.parameter_model[0]*k_tilde*self.value[1]) * torch.mean(curly_average(y**2))\
            + self.parameter_model[0]*k_tilde**3 * \
            self.value[1]*torch.mean(curly_average(y)**2)
        Q_tilde_hat = self.parameter_model[0]**2*k_tilde**2*self.value[1]\
            - 2*self.parameter_model[0]*k_tilde**3*self.value[1] * torch.mean(curly_average(y**2))\
            + k_tilde**2 * \
            (1+2*self.parameter_model[0]*k_tilde *
             self.value[1])*torch.mean(curly_average(y)**2)
        delta_o_hat = k_tilde**2 * \
            self.value[2] * (1/q-1)*(torch.mean(curly_average(y**2)
                                                )-torch.mean(curly_average(y)**2))

        log_exp = torch.max(H_tilde_prime, dim=1)[
                   0]+torch.log(torch.mean(exp_H_tilde_prime_alternative, dim=1))
        local_entropy = 1/2*self.value[1]*self.value[4]\
            - self.value[0]*self.value[3]\
            - self.value[2]*self.value[5]\
            - p_hat*self.parameter_model[2]**2\
            + 1/2*torch.log(k_tilde / (self.parameter_model[0]*self.parameter_model[3]))\
            - 1/2 * self.parameter_model[0] * k_tilde*self.value[1]\
            + torch.log(torch.sqrt(torch.tensor(2*torch.pi))*sigma_prime)\
            + torch.mean(log_exp)
        derivative = -2*self.parameter_model[2]*p_hat

        return (torch.tensor([q_tilde, Q_tilde, o, q_tilde_hat, Q_tilde_hat, delta_o_hat]), local_entropy, derivative)

    def update(self, value):
        self.value = self.parameter_iteration[1] * \
            self.value + (1-self.parameter_iteration[1]) * value

    def error(self):
        error_occur = False
        if self.value[0] < self.value[1]:
            print('q_tilde<Q_tilde')
            self.value[0] = self.value[1] + self.parameter_numeric[2]
            error_occur = True
        if self.value[1] < self.value[2]**2/self.parameter_numeric[0][0]:
            print('Q_tilde<o**2/q')
            self.value[1] = self.value[2]**2 / \
                self.parameter_numeric[0][0] + self.parameter_numeric[2]
            error_occur = True
        if self.value[4] < 0:
            print('Q_tilde_hat<0')
            self.value[4] = self.parameter_numeric[2]
            error_occur = True

        return error_occur

    def save(self):
        self.value_history.append(self.value.clone())
        self.local_entropy_history.append(self.local_entropy.clone())
        self.derivative_history.append(self.derivative)
        torch.save((self.value_history, self.local_entropy_history, self.derivative_history),
                   self.parameter_iteration[2]+'iteration.pt')

    def log(self, i):
        if (i + 1) % self.parameter_iteration[3] == 0:
            print(i, time.asctime(time.localtime(time.time())))
            print('local entropy:', self.local_entropy.item())
            print('derivate:', self.derivative.item())

    def is_converged(self, j):
        def mean(indicator, j):
            return torch.mean(torch.tensor(indicator)[int((j-1) * self.parameter_iteration[5]):int(j * self.parameter_iteration[5])])

        def absolute_difference(indicator):
            return torch.abs(mean(indicator, j)-mean(indicator, j-1))

        def relative_difference(indicator):
            return absolute_difference(indicator)/torch.abs(mean(indicator, j-1))

        return absolute_difference(self.local_entropy_history) < self.parameter_iteration[4][0] and (absolute_difference(self.derivative_history) < self.parameter_iteration[4][0] or relative_difference(self.derivative_history) < self.parameter_iteration[4][1])
