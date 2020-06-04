import torch
import numpy as np
from latent_functions import *
import random

# This file contains a set of helper functions called from experimental files. These functions mostly govern
# sampling from functions, creating contexts, standardizing data etc.


def sampleFromContext(context, latent_function, neural_dictionary, sample_range, samples = 20, chosen_value = None):

    if chosen_value != None:
        X_new = torch.tensor([chosen_value]).float()
    else:
        sample_min, sample_max = sample_range
        X_new = sampleX(samples, sample_min, sample_max)

    Y_new = sampleReward(latent_function, X_new)

    if context in neural_dictionary.context_dict:
        recalled_X = neural_dictionary.context_dict[context][2]
        recalled_Y = neural_dictionary.context_dict[context][3]

        X = torch.cat((recalled_X, X_new), 0)
        Y = torch.cat((recalled_Y, Y_new), 0)

    else:
        X = X_new
        Y = Y_new


    # scale_params_x = (torch.mean(X).item(), 1)
    # scale_params_y = (torch.mean(Y).item(), 1)

    # For certain kinds of data it's better to just subtract the mean from observations. This is because the GP
    # often just interprets variation as noise and ends up underfitting. When not standardized, the variance
    # becomes too pertinent to attribute to noise.

    #X_norm= standardize(X, scale_params_x[0], scale_params_x[1])
    #Y_norm= standardize(Y, given_mu=torch.mean(Y).item(), given_sd=1) # This is the best for y


    X_norm = X - torch.mean(X).item()
    Y_norm = Y - torch.mean(Y).item()

    return (X_new, X, Y)
    # return (X_new, X_norm, Y_norm, scale_params_x, scale_params_y, X, Y)


def standardize(X, given_mu = None, given_sd = None):

    if len(X) > 1:
        if given_mu == None and given_sd == None:
            mu_x = torch.mean(X).item()
            sd_x = torch.std(X).item()
            X = (X - mu_x)/sd_x
            return (X, mu_x, sd_x)
        else:
            X = (X - given_mu)/given_sd
            return X
    else:
        if given_mu == None:
            X = X - torch.mean(X).item()
        else:
            X = (X - given_mu) / given_sd
        return X
        #return (X, np.mean(X), 0)



def recover_unstandardized(X, mu_x, sd_X, discrete = False):

    X = (X * sd_X) + mu_x
    if discrete:
        X = X.int()

    return X


def sampleX(sample_size, x_min, x_max, uniform = False):
    if uniform:
        samples = torch.linspace(x_min, x_max, sample_size)
    else:
        samples = torch.tensor([random.uniform(x_min, x_max) for i in range(sample_size)])
    return samples


def sampleReward(latent_function, X):
    rewards = latent_function(X, sd = 0.1)
    return rewards


def createContext(random_function = False, trial = 0, context_n = 5):

    context_list = ["100", "010", "001", "110", "101", "011", "111"]
    all_functions = [linear, periodic, radial_basis, linear_periodic, linear_radial_basis,
                     radial_basis_periodic, lin_sin_rbf]



    context_list = [context_list[i] for i in range(context_n)]
    all_functions = [all_functions[i] for i in range(context_n)]


    if random_function:
        index = random.randint(0, len(context_list) - 1)
    elif context_n > 2:
        if trial < 2:
            index = 0
        elif trial < 4:
            index = 1
        elif trial < 6:
            index = 2
        else:
            index = random.randint(0, len(context_list) - 1)

    context = context_list[index]

    feature_vector = np.array([int(x) for x in context])
    latent_function = all_functions[index]

    return context, feature_vector, latent_function


def getPenalizedUniform(parameters, penalty=-0.35):
    # returns a prob distribution penalizing the number of optimizable parameters
    numerator = np.exp(penalty * parameters)
    denominator = np.sum(numerator)
    penalized_prior = numerator / denominator
    return penalized_prior


def getUniform(hypothesis_space, return_array = True):
    probability = 1/len(hypothesis_space)
    if return_array:
        arr = np.array([probability for i in range(len(hypothesis_space))])
        return arr
    else:
        return probability


def getGaussian(x, mu, sigma):
    return (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x - mu)/sigma)**2)

def get_kernel_params(kernels):
    parameter_arr = []
    for kernel in kernels:
        param_list = []
        for param_name, value in kernel.named_hyperparameters():
            param_list.append(param_name)

        parameter_arr.append(len(param_list))
    parameter_arr = np.array(parameter_arr)
    return parameter_arr