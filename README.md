# gaussian-process-rl
Bayesian GP models for contextual bandit tasks.

Code for performing a rough approximation of Automatic Bayesian Covariance Discovery (ABCD), as
well as code for a more Bayesian variant thereof: Both algorithms search through a space of kernels, giving 
possible explanations of the data. In ABCD, the kernel which maximizes the posterior likelihood (or minimizes BIC)
is selected to model the data, whereas in the more Bayesian variant, a composite GP is created, taking all explanations in the
kernel hypothesis space and their posterior probability into account when creating a predictive distribution.
Code for these algorithms can be found in the GP_inference.py file.



