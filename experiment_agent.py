import numpy as np
from matplotlib import pyplot as plt
import random
from GP_inference import *
from helper_functions import *
import NeuralDictionary
import Agent
import torch
from kernel_priors import *

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# This is an experiment testing bayesian automatic model construction with an RL agent which samples functions from
# different contexts. The RL agent encounters a context (a feature vector) and samples from a latent function and stores the
# resulting model in a dictionary, emulating episodic memories. These memories influence priors in later encounters with the same, or similar
# contexts. Furthermore, action values are computed from the GP model, and are used together with GP uncertainty
# to select actions. See Agent.py, NeuralDictionary and GP_inference for details.

# Create a set of kernels

lin = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(variance_prior=linear_variance_prior), outputscale_prior=linear_outputscale_prior)
cos = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(period_length_prior=per_period_len_prior), outputscale_prior=per_outputscale_prior)
rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=rbf_lengthscale_prior), outputscale_prior=rbf_outputscale_prior)

# create combinations of basis kernels. This space of kernels can be made arbitrarily large or small
lin_rbf_add = lin + rbf
lin_rbf_mul = lin * rbf
lin_cos_add = lin + cos
lin_cos_mul = lin * cos
rbf_cos_add = rbf + cos
rbf_cos_mul = rbf * cos
lin_cos_rbf_add = rbf + cos + lin

core_kernels = [lin, cos, rbf, lin_cos_add, lin_rbf_add, rbf_cos_add, lin_cos_rbf_add]
parameter_arr = get_kernel_params(core_kernels)

# get indices and assign a prior informed by kernel complexity
indices = np.array([i for i in range(len(core_kernels))])
penalized_prior = getPenalizedUniform(parameter_arr)

# set kernel params as penalized uniform
kernel_priors = penalized_prior

# Create neural dictionary containing episodic memories:
neuro_dict = NeuralDictionary.NeuralDictionary(indices, generalization_rate=2, kernel_parameters=parameter_arr)


# Define number of context samples
context_encounters = 150
# Define range of X


# Create the agent's representation of action space. This will consist of the possible X values it can sample
# during the bandit task. Initially, the expected value and confidence is zero for each action

action_space = torch.linspace(0, 10, 11)
action_mean, action_sd = torch.mean(action_space).item(), torch.std(action_space).item()

sample_range = (action_space.min().item(), action_space.max().item())
test_range = (sample_range[0] - action_sd, sample_range[1] + action_sd)

# Define how many samples the agent draws at each context encounter. In a bandit task setting, this should be 1.
# Then define number of test values
train_samples = 1
test_n = 100


initial_action_values = np.array([0 for i in range(len(action_space))])
initial_action_confidence = initial_action_values
# create agent
agent = Agent.Agent(action_space)


# Loop over encounters, select an action for each encounter and compute GP with observations
for i in range(context_encounters):
    # create a context, i.e. get context features and a reward function
    context, features, latent_function = createContext(random_function = True)
    # perform a lookup of episodic memories to inform prior
    kernel_scores = neuro_dict.look_up(features)
    # use kernel scores to get a new prior
    new_priors = neuro_dict.create_informed_prior(kernel_priors, kernel_scores)#kernel_priors#

    # get action values for the current context:
    if context in neuro_dict.context_dict:
        action_values = neuro_dict.get_action_values(context)
        action_confidence = neuro_dict.get_action_confidence(context)
    else:
        action_values, action_confidence = initial_action_values.copy(), initial_action_confidence.copy()

    # Make the agent choose an action based on UCB action values.
    transformed_action_values = agent.gpUpperConfidence(action_values, action_confidence)
    chosen_action = agent.stochastic_action(transformed_action_values, action_space)

    # Sample X, Y from latent function.
    data = sampleFromContext(context, latent_function, neuro_dict, sample_range, train_samples, chosen_action)
    # unpack tuple
    action, X, Y = data
    print(X)
    # create testing data, making sure that the action points are included in test_x
    test_x, action_indices = create_test_x(test_range, test_n, action_space)
    display = True if i > context_encounters - 140 else False


    # Perform bayesian inference over kernel space, and create posterior predicitive distribution which integrates over
    # posterior uncertainty
    posterior, predictive_dist, y_hat, uncertainty = bayesianGPInference(X, Y, core_kernels, new_priors, test_x=test_x, display_result=display)

    action_values, action_confidence = get_action_values(y_hat, uncertainty, action_indices)

    # Append context, data and posterior to dictionary, as well as action values and uncertainty
    neuro_dict.append(context, features, posterior, X, Y, action_values, action_confidence)

