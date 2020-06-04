import torch
import gpytorch
import GaussianProcessModel
import numpy as np
from matplotlib import pyplot as plt
from helper_functions import *


def bayesianGPInference(X, Y, kernels, kernel_priors, test_x, display_result = True):

    # Function for creating a GP model through Bayesian inference over possible GP models. This function serves as a
    # wrapper function, searching for different explanations of the data in a space of kernels, computes a posterior
    # over these kernel explanations, and constructs a composite predictive distribution, integrating over the possible
    # explanations.

    X, Y = X, Y  # training data must be tensors

    # Call function to obtain a set of models using the specified kernels, their marginal likelihoods, their predictive
    # distributions, and the mean of these distributions:
    model_list, likelihood_list, marginal_arr, predictive_dist_list, y_hat_list, lower, upper = computeGPyTorchMarginal(X, Y, kernels, test_x)
    # Get posterior from model evidence and priors:
    posterior = computeKernelPosterior(marginal_arr, kernel_priors)

    # Create a GP predictive distribution weighted by posterior beliefs about GP models
    y_hat, posterior_distribution, lower_confidence, upper_confidence = createMixtureGP(predictive_dist_list, y_hat_list, posterior, lower, upper)

    # Plot resulting GP
    if display_result:
        print(posterior)
        plotGP(X, Y, test_x, posterior_distribution, y_hat, lower_c = lower_confidence, upper_c = upper_confidence)

    return posterior, y_hat, upper_confidence


def computeGPyTorchMarginal(X, Y, kernel_space, test_x, display_each = False):
    # Function for computing log marginal likelihood for a set of kernels with ml II. The function creates a set
    # of resulting GP models, likelihood objects, an array of their log ML, their predictive distributions, and the
    # means of these distributions evaluated at all X in test_x
    marginal_array = []
    model_list = []
    likelihood_list = []
    predictive_dist_list = []
    y_hat_list = []
    lower_c, upper_c = [], []
    for kernel in kernel_space:
        model, likelihood, marginal_log_likelihood = makeGPyTorchModel(X, Y, kernel)
        model_list.append(model)
        marginal_likelihood = np.exp(marginal_log_likelihood)
        marginal_array.append(marginal_likelihood)
        likelihood_list.append(likelihood)

        predictive_dist = predict(test_x, likelihood, model)
        y_hat = predictive_dist.mean.detach().numpy()
        lower, upper = predictive_dist.confidence_region()

        predictive_dist_list.append(predictive_dist)
        y_hat_list.append(y_hat)
        # append confidence regions as np arrays
        lower_c.append(lower.numpy())
        upper_c.append(upper.numpy())

        if display_each:
            plotGP(X, Y, test_x, predictive_dist, y_hat)


    marginal_array = np.array(marginal_array)
    predictive_dist_list = np.array(predictive_dist_list)
    y_hat_list = np.array(y_hat_list)
    lower_c = np.array(lower_c)
    upper_c = np.array(upper_c)

    return model_list, likelihood_list, marginal_array, predictive_dist_list, y_hat_list, lower_c, upper_c


def predict(test_x, likelihood, model):
    # creates GP predictive distribution from a test data
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))

    return observed_pred



def createMixtureGP(predictive_distributions, y_hat_list, posterior, lower, upper):
    # creates a composite GP whose predictive distribution is a weighted sum of the posterior of GPs in hypothesis space
    weighted_predictives = predictive_distributions*posterior
    posterior_predictive = np.sum(weighted_predictives)
    y_hat = posterior_predictive.mean.detach().numpy()

    # lower_conf, upper_conf = posterior_predictive.confidence_region()
    # upper_conf = upper_conf.numpy()

    # get estimate of confidence
    adapted_posterior = posterior.reshape(-1, 1)
    weighted_lower, weighted_upper = lower*adapted_posterior, upper*adapted_posterior
    weighted_lower, weighted_upper = weighted_lower.sum(axis=0), weighted_upper.sum(axis=0)

    # predictions can also be obtained directly from the y_hat array:
    # weighted_means = y_hat_list * adapted_posterior
    # y_hat = weighted_means.sum(axis = 0)


    return y_hat, posterior_predictive, weighted_lower, weighted_upper


def computeKernelPosterior(ml_array, prior_array):
    # compute posterior with marginal and priors
    numerator = ml_array * prior_array
    denominator = np.sum(ml_array * prior_array)
    posterior = numerator/denominator

    return posterior

def makeGPyTorchModel(X, Y, kernel, learning_rate = 0.1):
    # Creates a gpytorch model from data with a specified kernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = GaussianProcessModel.GaussianProcessModel(X, Y, likelihood, kernel=kernel)
    marginal_log_likelihood = begin_training(model, X, Y, likelihood)

    return model, likelihood, marginal_log_likelihood


def begin_training(model, X, Y, likelihood, learning_rate=0.1, training_iterations=50, n_restarts = 3, optimize_noise = False):
    # Function for training GP with ML II:
    training_iter = training_iterations
    likelihood = likelihood

    # Set model and likelihood in train mode
    model.train()
    likelihood.train()

    # For the data we pass the GP models, we do not want to optimize noise, rather we want the model the see variances
    # as signals, and use these to optimize kernel hyperparameters. I.e. by telling the model that the signal to noise
    # ration is high, it will optimize ml w.r.t. kernel hyperparams.
    if not optimize_noise:
        model.likelihood.initialize(noise=0.1)
        likelihood.noise_covar.raw_noise.requires_grad_(False)

    n_restarts = n_restarts
    num_steps = training_iterations

    model_states = []
    losses = torch.zeros(n_restarts, num_steps)

    for i in np.arange(n_restarts):
        torch.random.initial_seed()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # "Loss" for GPs - the marginal log likelihood
        mll_ml = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


        for j in range(num_steps):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll_ml(output, Y)

            loss.backward()
            # if (j % num_steps == 0):
            #     print('Iter %d/%d - Loss: %.3f' % (j + 1, num_steps, loss.item()))
            optimizer.step()

            model_states.append({param_name: param.detach() for param_name, param in model.state_dict().items()})
            losses[i, j] = loss.item()

    best_final_idx = torch.argmin(losses[:, -1])
    best_model_params = model_states[best_final_idx]
    model.load_state_dict(best_model_params)

    # print(f'Actual outputscale: {model.covar_module.outputscale}')
    # print(f'Actual lengthscale: {model.covar_module.base_kernel.lengthscale}')
    #print(f'Actual period length: {model.covar_module.base_kernel.period_length}')

    # compute final mll
    output = model(X)
    marginal_log_likelihood = mll_ml(output, Y).item()
    print(marginal_log_likelihood)
    return marginal_log_likelihood

def plotGP(train_x, train_y, test_x, predictive_dist, y_hat, plot_extra_confidence = True, lower_c = None, upper_c = None):
    # plots a GPs predictive distribution and confidence region
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Get upper and lower confidence bounds
        lower, upper = predictive_dist.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), y_hat, 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        if plot_extra_confidence:
            ax.fill_between(test_x.numpy(), lower_c, upper_c, alpha=0.25)
        ax.set_ylim([np.min(y_hat) - 2, np.max(y_hat) + 2])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()


def create_test_x(test_range, test_n, action_space = None):
    test_min, test_max = test_range
    test_n = test_n

    test_x = torch.linspace(test_min, test_max, test_n)

    # If there's a set of actions, make sure to include them in the test_x
    if action_space != None:
        test_x = torch.cat((test_x, action_space), 0)

        old_indices = np.array([i for i in range(len(test_x) - len(action_space), len(test_x))])
        test_x, indices = torch.sort(test_x)
        indices = indices.numpy()

        new_indices = []
        for index in old_indices:
            loc = np.where(indices == index)
            new_indices.append(loc)

        new_indices = np.array(new_indices).flatten()
        return test_x, new_indices
    else:
        return test_x

def get_action_values(y_hat, uncertainty, action_indices):
    action_values = y_hat[action_indices]
    action_uncertainty = uncertainty[action_indices] - action_values  # this gives the standard dev at each action point
    return action_values, action_uncertainty


def ABCD(X, Y, kernels, kernel_priors, test_x):
    # A rough approximation of Duvenaud's ABCD. Instead of adding kernels incrementally, this procedure simply considers
    # a subset of kernels from the kernel grammar, each with a prior probability, and selects whichever kernel has
    # the highest posterior. If prior reflects model complexity, this becomes roughly equivalent to minimizing BIC.
    X, Y = X, Y
    #
    # test_min, test_max, test_n = X.min().item(), X.max().item(), len(X)
    # #test_x = torch.linspace(test_min, test_max, test_n)

    model_list, likelihood_list, marginal_arr, predictive_dist_list, y_hat_list, lc, uc = computeGPyTorchMarginal(X, Y, kernels,
                                                                                                          test_x)
    posterior = computeKernelPosterior(marginal_arr, kernel_priors)
    max_posterior_index = np.argmax(posterior)
    max_posterior_model = model_list[max_posterior_index]
    max_posterior_likelihood = likelihood_list[max_posterior_index]
    max_posterior_distribution = predictive_dist_list[max_posterior_index]
    max_posterior_yhat = y_hat_list[max_posterior_index]


    plotGP(X, Y, test_x, max_posterior_distribution, max_posterior_yhat, plot_extra_confidence=False)

    return max_posterior_model, max_posterior_likelihood, posterior

def BIC(log_ml, n, params):
    # compute BIC
    bic = (params * np.log(n)) - (2 * log_ml)
    return bic






