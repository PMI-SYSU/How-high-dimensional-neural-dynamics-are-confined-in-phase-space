from utils import *


class finder:
    def __init__(self, g, N, epsilon):
        self.g = g
        self.N = N
        self.epsilon = epsilon
        J = self.g / np.sqrt(self.N) * np.random.randn(self.N, self.N)
        np.fill_diagonal(J,0)
        np.save('J.npy', J)
        self.v = lambda x: -x + J @ np.tanh(x)
        self.n_c = 1
        self.x = np.zeros(self.N)[np.newaxis, :]
        self.n_u = 1

    def main(self, step, std, tolerance, proportion, data_path, log_interval):
        print(time.asctime(time.localtime(time.time())), 'start')
        for i in range(step):
            x0 = np.random.normal(0, std, self.N)
            x_c = least_squares(self.v, x0, method='lm').x
            if np.linalg.norm(self.v(x_c))/np.sqrt(self.N) < self.epsilon:
                self.n_c += 1
                if np.min(np.linalg.norm(x_c-self.x, axis=1)/np.sqrt(self.N)) > tolerance:
                    self.x = np.vstack((self.x, x_c))
                    self.n_u = len(self.x)
            if self.n_u/self.n_c < proportion:
                break
            self.save(i, data_path)
            self.log(i, log_interval, step)
        print(time.asctime(time.localtime(time.time())), 'end')

    def save(self, i, data_path):
        with FileLock(data_path+'finder.lock'):
            if i == 0:
                with h5py.File(data_path+'finder.h5', 'w') as f:
                    f.create_dataset('x', data=self.x, maxshape=(None, None))
                    f.create_dataset('n_c', data=[self.n_c], maxshape=(None,))
                    f.create_dataset('n_u', data=[self.n_u], maxshape=(None,))
            else:
                with h5py.File(data_path+'finder.h5', 'a') as f:
                    x = f['x']
                    x.resize((self.x.shape[0], self.x.shape[1]))
                    x[:] = self.x
                    n_c = f['n_c']
                    n_c.resize((n_c.shape[0]+1,))
                    n_c[-1] = self.n_c
                    n_u = f['n_u']
                    n_u.resize((n_u.shape[0]+1,))
                    n_u[-1] = self.n_u

    def log(self, i, log_interval, step):
        if (i + 1) % log_interval == 0:
            print(time.asctime(time.localtime(time.time())), str(round((i+1)/step * 1e2, 1)) +
                  '%')
