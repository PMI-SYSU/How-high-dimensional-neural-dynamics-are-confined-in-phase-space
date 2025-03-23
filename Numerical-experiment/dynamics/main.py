import os
import sys
sys.path.append(
    '/Users/shihsherwang/Local/Github/RNN-fixed-point-distribution/dynamics')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameter import *
from trajectory import *

print(time.asctime(time.localtime(time.time())), 'start')
for i in range(len(x)):
    t = 100
    RNN = trajectory(g, N, Delta_t, 'RNN', x=x[i], epsilon=epsilon, J=J, t=t)
    step = RNN.main(dir+'/RNN_g_'+str(g)+'_'+str(i)+'_')
    Langevin = trajectory(g, N, Delta_t, 'Langevin', T=T, x=x[i], J=J)
    Langevin.main(dir+'/Langevin_g_'+str(g)+'_T_' +
                  str(T)+'_'+str(i)+'_', step=step)
    if (i + 1) % log_interval == 0:
        print(time.asctime(time.localtime(time.time())), str(round((i+1)/len(x) * 1e2, 1)) +
                '%')
print(time.asctime(time.localtime(time.time())), 'end')