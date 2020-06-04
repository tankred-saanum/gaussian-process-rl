import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import numpy as np
from GP_inference import *
from latent_functions import *
from helper_functions import *
from kernel_priors import *


#This is an experiment for testing automatic GP models on a non-linear function.
#Here a rough approx of Duvenaud's ABCD is compared to a more Bayesian automatic model construction procedure.
# ABCD is performed by calling the ABCD function with the data and kernels, whereas the more Bayesian variant is
# performed by calling the BayesianGPInference function. See GP_inference for details

min_x, max_x, n = 0, 2, 100
train_x = torch.linspace(min_x, max_x, n)
# train_y = linear(train_x, sd = 0.01) + radial_basis(train_x) + linear_periodic(train_x)
train_y = random_nonlinear(train_x, sd= 0.1)

test_x = create_test_x((-1, 3), 200)
# Create a set of kernels

lin = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(variance_prior=linear_variance_prior), outputscale_prior=linear_outputscale_prior)
cos = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(period_length_prior=per_period_len_prior), outputscale_prior=per_outputscale_prior)
rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=rbf_lengthscale_prior), outputscale_prior=rbf_outputscale_prior)


lin_rbf_add = lin + rbf
lin_rbf_mul = lin * rbf
lin_cos_add = lin + cos
lin_cos_mul = lin * cos
rbf_cos_add = rbf + cos
rbf_cos_mul = rbf * cos
lin_cos_rbf_add = rbf + cos + lin

core_kernels = [lin, cos, rbf, lin_cos_add, lin_rbf_add, rbf_cos_add, lin_cos_rbf_add]
parameter_arr = get_kernel_params(core_kernels)


penalized_prior = getPenalizedUniform(parameter_arr)
#set kernel params as penalized uniform
kernel_priors = penalized_prior

# perform abcd:
model, likelihood, posterior = ABCD(train_x, train_y, core_kernels, kernel_priors, test_x)
# create a more bayesian composite model:
posterior, y_hat, confidence = bayesianGPInference(train_x, train_y, core_kernels, kernel_priors, test_x)
