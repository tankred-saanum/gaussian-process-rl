import torch
import numpy as np
import gpytorch
import random
from kernel_priors import *

class KernelFamily():
    def __init__(self, basis_kernels, starting_kernels, complexity_penalty = 0.5, only_additive = False):
        self.basis_kernels = basis_kernels
        self.kernel_bases = [type(kern.base_kernel) for kern in self.basis_kernels]
        self.starting_kernels = starting_kernels
        self.complexity_pen = complexity_penalty
        self.normalizing_constant = 1/len(self.basis_kernels)
        self.only_additive = only_additive


    def flip(self):
        self.proposal = random.uniform(0, 1)
        self.accept = True if self.proposal > self.complexity_pen else False
        return self.accept


    def get_kernel_likelihood(self, kernel):
        # get estimate of the likelihood of a single kernel
        self.base = []
        self.get_base_kernel_types(kernel, self.base)

        self.base = np.array(self.base)

        if set(self.base).issubset(self.kernel_bases):

            return self.complexity_pen * (1/(len(self.basis_kernels)*2))** (np.abs(len(self.basis_kernels) - len(self.base)) + 1)
        else:
            return 0

    def get_kernel_space_likelihoods(self, kernel_space):
        # get estimate of likelihoods of a set of kernels
        self.likelihood_arr = []
        for kernel in kernel_space:
            self.base = []
            self.get_base_kernel_types(kernel, self.base)

            if set(self.base) == set(self.kernel_bases) and len(self.base) == len(self.kernel_bases): #set(self.base).issubset(self.kernel_bases) and set(self.kernel_bases).issubset(self.base):

                self.likelihood = self.complexity_pen
                self.likelihood_arr.append(self.likelihood)

            elif set(self.base).issubset(self.kernel_bases):

                self.likelihood = (1 - self.complexity_pen) * (1/(len(self.basis_kernels)*2))**(np.abs(len(self.basis_kernels) - len(self.base)))
                self.likelihood_arr.append(self.likelihood)

            else:
                self.likelihood = 0
                self.likelihood_arr.append(self.likelihood)

        self.likelihood_arr = np.array(self.likelihood_arr)

        return self.likelihood_arr




    def get_base_kernel_types(self, kernel, base_list):
        # recursive method for getting the basis kernels of any given kernel, be it compositional or not
        # appends base kernels to the list passed as base_list
        if hasattr(kernel, "base_kernel"):
            base_list.append(type(kernel.base_kernel))
        else:
            for sub_kern in kernel.kernels:
                self.get_base_kernel_types(sub_kern, base_list)


    def give_hyperpriors(self):
        pass
        # TODO


    def create_kernel_space(self, root_kernel):
        self.kernel_space = []

        for bk in self.basis_kernels:

            if self.only_additive:
                self.additive_kernel = root_kernel + bk
                self.kernel_space.append(self.additive_kernel)

            else:
                self.additive_kernel = root_kernel + bk
                self.kernel_space.append(self.additive_kernel)
                self.mult_kernel = root_kernel * bk

                self.kernel_space.append(self.mult_kernel)

        return self.kernel_space

    def draw_multiple(self, num_draws, unique = True, includes_root = True):

        self.kernel_draws = []
        self.marginal_l = []
        self.num_draws = num_draws

        if includes_root:
            self.kernel_idx = random.randint(0, len(self.starting_kernels) - 1)
            self.kernel = self.starting_kernels[self.kernel_idx]
            self.kernel_prob = (1 / len(self.starting_kernels)) * (self.complexity_pen)
            self.kernel_draws.append(self.kernel)
            self.marginal_l.append(self.kernel_prob)
            self.num_draws -= 1

        for i in range(self.num_draws):
            self.drawn_k, self.ml = self.draw()

            if unique:
                while self.drawn_k in self.kernel_draws:
                    self.drawn_k, self.ml = self.draw()

            self.kernel_draws.append(self.drawn_k)
            self.marginal_l.append(self.ml)

        self.kernel_draws = np.array(self.kernel_draws)
        self.marginal_l = np.array(self.marginal_l)
        return self.kernel_draws, self.marginal_l


    def draw(self, kernel_space = None, depth = 1):

        self.hypothesis_space = self.starting_kernels if kernel_space == None else kernel_space
        self.depth = depth

        self.kernel_idx = random.randint(0, len(self.hypothesis_space) - 1)
        self.current_kernel = self.hypothesis_space[self.kernel_idx]

        self.kernel_prob = (self.complexity_pen * (1/len(self.hypothesis_space)))**self.depth

        if self.flip():
            self.depth += 1
            self.kernel_space = self.create_kernel_space(self.current_kernel)
            self.current_kernel, self.kernel_prob = self.draw(self.kernel_space, depth=self.depth)

        return self.current_kernel, self.kernel_prob
