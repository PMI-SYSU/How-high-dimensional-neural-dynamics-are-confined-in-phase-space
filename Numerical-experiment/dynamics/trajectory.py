from utils import *


class trajectory:
    def __init__(self, g, N, Delta_t, dynamics, t=None, T=None, x=None, J=None, epsilon=None):
        self.g = g
        self.N = N
        self.Delta_t = Delta_t
        self.dynamics = dynamics
        self.t = t
        self.T = T
        if x is not None:
            self.x = torch.tensor(x.astype(np.float32))
        else:
            self.x = torch.randn(self.N)
        if J is not None:
            self.J = torch.tensor(J.astype(np.float32))
        else:
            self.J = self.g/np.sqrt(self.N) * torch.randn(self.N, self.N)
            self.J.fill_diagonal_(0)
        self.epsilon = epsilon

    def main(self, data_path, log_interval=None, step=None):
        if log_interval is not None:
            print(time.asctime(time.localtime(time.time())), 'start')
        if step is None:
            step = int(self.t/self.Delta_t)
        if self.dynamics == 'RNN':
            f = self.RNN
        elif self.dynamics == 'Langevin':
            f = self.Langevin
        self.v = f(self.x)
        self.save(data_path, new_file=True)
        for i in range(step):
            self.v = f(self.x)
            if self.epsilon is not None:
                if torch.norm(self.v)/torch.sqrt(torch.tensor(self.N)) > self.epsilon:
                    i -= 1
                    break
            Delta_x = self.v * self.Delta_t
            self.x = self.x + Delta_x
            self.save(data_path)
            if log_interval is not None:
                self.log(i, log_interval, step)
        if log_interval is not None:
            print(time.asctime(time.localtime(time.time())), 'end')
        return i+1

    def RNN(self, x):
        def phi(x): return torch.tanh(x)
        return -x + self.J @ phi(x)

    def Langevin(self, x):
        def phi(x): return torch.tanh(x)
        def phi_prime(x): return 1-torch.tanh(x)**2
        h = self.J@phi(x)
        epsilon = torch.randn(self.N)
        return -x+h-phi_prime(x)*self.J.T@(h-x)+torch.sqrt(torch.tensor(2*self.T/self.Delta_t))*epsilon

    def save(self, data_path, new_file=False):
        with FileLock('trajectory.lock'):
            if new_file:
                with h5py.File(data_path+'trajectory.h5', 'w') as f:
                    f.create_dataset(
                        'x', data=[self.x.cpu().numpy(),], maxshape=(None, None))
                    f.create_dataset(
                        'v', data=[self.v.cpu().numpy(),], maxshape=(None, None))
            else:
                with h5py.File(data_path+'trajectory.h5', 'a') as f:
                    x = f['x']
                    x.resize((x.shape[0]+1, x.shape[1]))
                    x[-1] = self.x.cpu().numpy()
                    v = f['v']
                    v.resize((v.shape[0]+1, v.shape[1]))
                    v[-1] = self.v.cpu().numpy()

    def log(self, i, log_interval, step):
        if (i + 1) % log_interval == 0:
            print(time.asctime(time.localtime(time.time())), str(round((i+1)/step * 1e2, 1)) +
                  '%')
