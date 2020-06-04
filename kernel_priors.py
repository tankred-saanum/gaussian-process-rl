import gpytorch


rbf_outputscale_prior = gpytorch.priors.GammaPrior(15, 1)
linear_outputscale_prior = gpytorch.priors.GammaPrior(4, 1)
per_outputscale_prior = gpytorch.priors.GammaPrior(2, 1)
rbf_lengthscale_prior = gpytorch.priors.GammaPrior(8, 3)
per_lengthscale_prior = rbf_lengthscale_prior  
per_period_len_prior = gpytorch.priors.GammaPrior(6, 3)
linear_variance_prior = gpytorch.priors.LogNormalPrior(0, 0.02)
