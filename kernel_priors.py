import gpytorch


rbf_outputscale_prior = gpytorch.priors.GammaPrior(15, 1)#GammaPrior(8, 0.5)#LogNormalPrior(3, 2.5)#gpytorch.priors.GammaPrior(1.5, 2)
linear_outputscale_prior = gpytorch.priors.GammaPrior(4, 1)#(1.5, 0.15)#LogNormalPrior(4, 0.8)
per_outputscale_prior = gpytorch.priors.GammaPrior(2, 1)#GammaPrior(1.5, 0.15)#GammaPrior(1.5, 0.15)this one worked well#LogNormalPrior(2, 1)
rbf_lengthscale_prior = gpytorch.priors.GammaPrior(8, 3)#GammaPrior(4., 2.)
per_lengthscale_prior = rbf_lengthscale_prior  # outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
per_period_len_prior = gpytorch.priors.GammaPrior(6, 3)#GammaPrior(1.5, 0.2)#LogNormalPrior(0, .025)  #UniformPrior(0.1, 2.5)   #
linear_variance_prior = gpytorch.priors.LogNormalPrior(0, 0.02)