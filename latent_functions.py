import torch
import math
import random
from matplotlib import pyplot as plt
import numpy as np

# Functions adapted to take tensor inputs and output y as a tensor

def periodic(x, sd = 0.1):
    return (torch.sin(x * (0.6 * math.pi)) + gaussianNoise(x, sd))*1.2

def jigsaw(x, beta = 10.0, sd = 0.1):
    x_new = x.clone()
    x_new[x % 2 == 0] = beta
    x_new[x % 2 != 0] = -beta
    return x_new

def linear(x, beta = 0.6, sd = 0.2):
    return (x * beta) + gaussianNoise(x, sd)

def linear_periodic(x, beta = 0.8, sd= 0.2):
    new_sd = sd/2
    return linear(x, beta, new_sd) + periodic(x, new_sd)

def radial_basis(x, beta = 15, lengthscale = 2, optimum = 5, sd = 0.2):
    if optimum == None:
        optimum = (x.max().item() - x.min().item())/2
    else:
        optimum = optimum
    return beta * (torch.exp(-((optimum-x)**2)/(2*lengthscale)**2)) + gaussianNoise(x, sd)

def linear_radial_basis(x, beta1= 0.8, beta2 = 3, lengthscale = 2, optimum = None, sd = 0.2):
    new_sd = sd / 2
    return linear(x, beta1, new_sd) + radial_basis(x, beta2, lengthscale, optimum, new_sd)


def lin_sin_rbf(x, beta1= 0.8, beta2 = 3, lengthscale = 2, optimum = None, sd = 0.2):
    new_sd = sd/3
    return linear(x, beta1, new_sd) + radial_basis(x, beta2, lengthscale, optimum, new_sd) + periodic(x, new_sd)

def radial_basis_periodic(x, beta = 3, lengthscale = 2, optimum = None, sd = 0.2):
    new_sd = sd/2
    return periodic(x, new_sd) + radial_basis(x, beta, lengthscale, optimum, new_sd)

def random_nonlinear(x, sd = 0.2):
    o1, o2, o3 = (x.max().item() - x.min().item())*0.2, (x.max().item() - x.min().item())*0.4, (x.max().item() - x.min().item())*0.6
    l1, l2, l3 = random.uniform(0, 3), random.uniform(0, 3), random.uniform(0, 3)
    b1, b2, b3 = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
    b1, b2, b3 = -b1 if random.randint(0, 1) else b3, -b2 if random.randint(0, 1) else b3, -b3 if random.randint(0, 1) else b3
    return radial_basis(x, b1, l1, o1, sd) + radial_basis(x, b2, l2, o2, sd) + radial_basis(x, b3, l3, o3, sd)

def gaussianNoise(x, sd):
    return torch.randn(x.size()) * sd


def linear_jigsaw(x, beta1 = 0.8, beta2 = 4., sd = 0.1):
    return linear(x, beta1, sd) + jigsaw(x, beta2, sd)
