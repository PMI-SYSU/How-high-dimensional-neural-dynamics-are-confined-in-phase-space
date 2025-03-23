import warnings
import scipy.optimize as optimize
import numpy as np
import itertools
import time
import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')
elif torch.backends.mps.is_available():
    torch.set_default_device('mps')


def phi(x): return torch.tanh(x)


def golden_section_search(f, a, b, precision):
    golden_ratio = (1+torch.sqrt(torch.tensor(5)))/2
    while torch.all(torch.abs(b-a) > precision):
        a_prime = b-(b-a)/golden_ratio
        b_prime = a+(b-a)/golden_ratio
        a_less_b = f(a_prime) < f(b_prime)
        b[a_less_b] = b_prime[a_less_b]
        a[~a_less_b] = a_prime[~a_less_b]
    x = (a+b)/2
    return (x, f(x))


warnings.filterwarnings("ignore")
