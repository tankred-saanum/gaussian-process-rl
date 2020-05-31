import torch
import gpytorch
import GaussianProcessModel
import numpy as np
from matplotlib import pyplot as plt


def bayesianGPInference(X, Y, kernels, kernel_priors):

    # Function for creating a GP model through Bayesian inference over possible GP models. This function serves as a
    # wrapper function, searching for different explanations of the data in a space of kernels, computes a posterior
    # over these kernel explanations, and constructs a composite predictive distribution, integrating over the possible
    # explanations.


    X, Y = X, Y  # training data must be tensors

    # Create test data in range of training data to illustrate resulting predictive distribution
    test_min, test_max, test_n = X.min().item(), X.max().item(), len(X)
    test_x = torch.linspace(test_min, test_max, test_n)

    # Call function to obtain a set of models using the specified kernels, their marginal likelihoods, their predictive
    # distributions, and the mean of these distributions:
    model_list, likelihood_list, marginal_arr, predictive_dist_list, y_hat_list = computeGPyTorchMarginal(X, Y, kernels, test_x)
    # Get posterior from model evidence and priors:
    posterior = computeKernelPosterior(marginal_arr, kernel_priors)

    # Create a GP predictive distribution weighted by posterior beliefs about GP models
    y_hat, posterior_distribution = createMixtureGP(predictive_dist_list, y_hat_list, posterior)

    # Plot resulting GP
    plotGP(X, Y, test_x, posterior_distribution, y_hat)

    # return posterior
    return posterior


def computeGPyTorchMarginal(X, Y, kernel_space, test_x):
    # Function for computing log marginal likelihood for a set of kernels with ml II. The function creates a set
    # of resulting GP models, likelihood objects, an array of their log ML, their predictive distributions, and the
    # means of these distributions evaluated at all X in test_x
    marginal_array = []
    model_list = []
    likelihood_list = []
    predictive_dist_list = []
    y_hat_list = []
    for kernel in kernel_space:
        model, likelihood, marginal_log_likelihood = makeGPyTorchModel(X, Y, kernel)
        model_list.append(model)
        marginal_likelihood = np.exp(marginal_log_likelihood)
        marginal_array.append(marginal_likelihood)
        likelihood_list.append(likelihood)

        predictive_dist = predict(test_x, likelihood, model)
        y_hat = predictive_dist.mean.detach().numpy()
        predictive_dist_list.append(predictive_dist)
        y_hat_list.append(y_hat)


    marginal_array = np.array(marginal_array)
    predictive_dist_list = np.array(predictive_dist_list)
    y_hat_list = np.array(y_hat_list)

    return model_list, likelihood_list, marginal_array, predictive_dist_list, y_hat_list


def predict(test_x, likelihood, model):
    # creates GP predictive distribution from a test data
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))

    return observed_pred


def createMixtureGP(predictive_distributions, y_hat_list, posterior):
    # creates a composite GP whose predictive distribution is a weighted sum of the posterior of GPs in hypothesis space
    weighted_predictives = predictive_distributions*posterior
    posterior_predictive = np.sum(weighted_predictives)
    y_hat_new = posterior_predictive.mean.detach().numpy()

    # this can also be done directly from the y_hat array:
    # adapted_posterior = posterior.reshape(-1, 1)
    # weighted_means = y_hat_list * adapted_posterior
    # y_hat = weighted_means.sum(axis = 0)


    return y_hat_new, posterior_predictive


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


def begin_training(model, X, Y, likelihood, learning_rate = 0.1, training_iterations = 50):
        # Function for training GP with ML II:
        training_iter = training_iterations
        likelihood = likelihood

        # Set model and likelihood in train mode
        model.train()
        likelihood.train()

        # use adam algorithm
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr=learning_rate)

        # loss = negative log marginal likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, Y)
            loss.backward()
            print("iteration and loss: ",
                i + 1, training_iter, loss.item(),

            )

            optimizer.step()

        # compute final mll
        output = model(X)
        marginal_log_likelihood = mll(output, Y).item()

        return marginal_log_likelihood

def plotGP(train_x, train_y, test_x, predictive_dist, y_hat):
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
        ax.set_ylim([np.min(y_hat) - 2, np.max(y_hat) + 2])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()



def ABCD(X, Y, kernels, kernel_priors):
    # A rough approximation of Duvenaud's ABCD. Instead of adding kernels incrementally, this procedure simply considers
    # a subset of kernels from the kernel grammar, each with a prior probability, and selects whichever kernel has
    # the highest posterior. If prior reflects model complexity, this becomes roughly equivalent to minimizing BIC.
    X, Y = X, Y

    test_min, test_max, test_n = X.min().item(), X.max().item(), len(X)
    test_x = torch.linspace(test_min, test_max, test_n)

    model_list, likelihood_list, marginal_arr, predictive_dist_list, y_hat_list = computeGPyTorchMarginal(X, Y, kernels,
                                                                                                          test_x)
    posterior = computeKernelPosterior(marginal_arr, kernel_priors)
    max_posterior_index = np.argmax(posterior)
    max_posterior_model = model_list[max_posterior_index]
    max_posterior_likelihood = likelihood_list[max_posterior_index]
    max_posterior_distribution = predictive_dist_list[max_posterior_index]
    max_posterior_yhat = y_hat_list[max_posterior_index]


    plotGP(X, Y, test_x, max_posterior_distribution, max_posterior_yhat)

    return max_posterior_model, max_posterior_likelihood, posterior

def BIC(log_ml, n, params):
    # compute BIC
    bic = (params * np.log(n)) - (2 * log_ml)
    return bic


