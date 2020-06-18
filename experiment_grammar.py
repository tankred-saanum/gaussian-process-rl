import numpy as np
from matplotlib import pyplot as plt
import random
from GP_inference import *
from helper_functions import *
import NeuralDictionary
import gpytorch
from kernel_priors import *
import KernelFamily


float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Experiment for compositional transfer learning. See the ipynb notebook for a detailed demonstration.

lin = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(variance_prior=linear_variance_prior), outputscale_prior=linear_outputscale_prior)
cos = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(period_length_prior=per_period_len_prior), outputscale_prior=per_outputscale_prior)
rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=rbf_lengthscale_prior), outputscale_prior=rbf_outputscale_prior)

lin_rbf_add = lin + rbf
lin_rbf_mul = lin * rbf
lin_cos_add = lin + cos
lin_cos_mul = lin * cos
rbf_cos_add = rbf + cos
rbf_cos_mul = rbf * cos
cos_rbf_add = cos + rbf
cos_rbf_mul = cos * rbf
lin_cos_rbf_add = rbf + cos + lin

##############
# define kernel families:
lin_root = [lin]
lin_core = lin_root
lin_familiy = KernelFamily.KernelFamily(lin_core, lin_root, only_additive=True)


cos_root = [cos]
cos_core = cos_root
cos_familiy = KernelFamily.KernelFamily(cos_core, cos_root)


rbf_root = [rbf]
rbf_core = rbf_root
rbf_family = KernelFamily.KernelFamily(rbf_core, rbf_root)


# lin_cos_root = [lin_cos_add, lin_cos_mul]
lin_cos_root = [lin_cos_add]
lin_cos_core = [lin, cos]
lin_cos_family = KernelFamily.KernelFamily(lin_cos_core, lin_cos_root)


# lin_rbf_root = [lin_rbf_add, lin_rbf_mul]
lin_rbf_root = [lin_rbf_add]
lin_rbf_core = [lin, rbf]
lin_rbf_family = KernelFamily.KernelFamily(lin_rbf_core, lin_rbf_root)


rbf_cos_root = [cos_rbf_add]
#rbf_cos_root = [cos_rbf_add, cos_rbf_mul]
rbf_cos_core = [cos, rbf]
cos_rbf_family = KernelFamily.KernelFamily(rbf_cos_core, rbf_cos_root)

lin_cos_rbf_root = [lin_cos_rbf_add]
lin_cos_rbf_core = [lin, cos, rbf]
lin_cos_rbf_family = KernelFamily.KernelFamily(lin_cos_rbf_core, lin_cos_rbf_root)

#########################################

kernel_families = [lin_familiy, cos_familiy, rbf_family, lin_cos_family, lin_rbf_family, cos_rbf_family, lin_cos_rbf_family]
family_complexity = np.array([len(fam.basis_kernels) for fam in kernel_families])


family_priors = getPenalizedUniform(family_complexity).reshape(-1, 1)

core_kernels, probabilities = sample_from_grammar_fragments(kernel_families, family_priors, num_draws=1, includes_root=True)


parameters = get_kernel_params(core_kernels)

indices = np.array([i for i in range(len(core_kernels))])



#Create neural dictionary containing episodic memories:
neuro_dict = NeuralDictionary.NeuralDictionary(indices, generalization_rate=1, kernel_parameters=parameters)

# Define number of context samples
context_encounters = 30
learning_encounters = 3
# Define range of X
sample_range = (0, 15)
sample_mean, sample_sd = 5, 3.3
test_range = (sample_range[0] - sample_sd, sample_range[1] + sample_sd)

# Define how many samples the agent draws at each context encounter. In a bandit task setting, this should be 1.
# Then define number of test values
train_samples = 10
test_samples = 100

headers = ["lin", "cos", "rbf", "lin_cos", "lin_rbf", "cos_rbf", "lin_cos_rbf"]
for i in range(context_encounters):
    # create a context, i.e. get context features and a reward function
    context, features, latent_function = createContext(trial=i)
    # perform a lookup of episodic memories to inform prior

    kernel_transfer_prior = neuro_dict.bayesian_transfer(context, features, probabilities)

    ###############
    # This snippet of code computes the transfer prior

    family_transfer_prior = compute_family_posterior(kernel_families, family_priors, core_kernels)
    joint, family_transfer_prior = compute_joint_posterior(kernel_transfer_prior, family_transfer_prior)

    new_family_prior = family_transfer_prior.reshape(-1, 1)
    core_kernels, kernel_priors = sample_from_grammar_fragments(kernel_families, new_family_prior)

    test2 = np.sum(kernel_priors)

    ###############

    # Sample X, Y from latent function
    data = sampleFromContext(context, latent_function, neuro_dict, sample_range, train_samples)
    action, train_x, train_y = data

    test_x = create_test_x((train_x.min().item() - 3, train_x.max().item()+ 3), 200)




    posterior, predictive_dist, y_hat, confidence = bayesianGPInference(train_x, train_y, core_kernels, kernel_priors, test_x, display_result=False)

    print("test2, ", test2)
    print("kernel priors ", kernel_priors)
    print("kernel transfer ", kernel_transfer_prior)
    print(headers)
    print("family ", family_transfer_prior.reshape(1, -1), "\n")

    #print(new_family_prior.flatten())
    print("posterior ", posterior)
    print(context)
    plotGP(train_x, train_y, test_x, predictive_dist, y_hat, plot_extra_confidence=False)
    # Append context, data and posterior to dictionary
    neuro_dict.append(context, features, posterior, train_x, train_y)




