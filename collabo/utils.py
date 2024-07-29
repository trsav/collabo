import json
import warnings
import numpy as np
import scipy.optimize as opt
import gpytorch
import torch
from tqdm import tqdm
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize as minimize_mo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.config import Config
from typing import Optional
import matplotlib.pyplot as plt

Config.warnings["not_compiled"] = False


def distribute_solutions(fixed_solutions: Optional[np.ndarray], bounds: np.ndarray, required: int):
    """A small optimization problem to optimally space solutions around the expert solutions. 

    `required` solutions are distributed around the `fixed_solutions` in the solution space defined by `bounds`.
    If there are no fixed solutions, the solutions are distributed using Latin Hypercube Sampling.

    :param fixed_solutions: Expert-defined solutions to be used as fixed points within the solution space.
    :type fixed_solutions: np.ndarray, optional
    :param bounds: The bounds of the solution space.
    :type bounds: np.ndarray
    :param required: The number of solutions to distribute.
    :type required: int
    """

    # normalize the fixed solutions
    fixed_solutions = (fixed_solutions - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    def obj(remaining_solutions, fixed_solutions):
        remaining_solutions = np.reshape(remaining_solutions, (required, len(bounds)))
        x_aug = np.concatenate((remaining_solutions, fixed_solutions), axis=0)
        n = len(x_aug)
        K = np.zeros((n, n))  # preallocate covariance matrix
        for i in range(n):
            for j in range(i, n):
                # basic distance metric
                K[i, j] = np.exp(-np.sum((x_aug[i] - x_aug[j]) ** 2))
                K[j, i] = K[i, j]
        # determinant of the covariance matrix
        det = np.linalg.det(K)
        return -det

    remaining_solutions = np.random.uniform(
        size=(required * len(bounds))
    )  # random initial guess

    print(f"Distributing remaining {required} solutions")

    res = opt.minimize(
        obj,
        remaining_solutions,
        args=(fixed_solutions),
        method="Nelder-Mead",
        bounds=[[0, 1] for i in range(required * len(bounds))],
        options={"disp": True, "maxiter": 100000},
    )
    optimal_solutions = np.reshape(res.x, (required, len(bounds)))
    # add the fixed solutions back in
    optimal_solutions = np.concatenate((optimal_solutions, fixed_solutions), axis=0)
    # unnormalize the solutions
    optimal_solutions = optimal_solutions * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    return optimal_solutions.tolist()


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=len(train_x[0]))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, noise=None, gp_iter=1000):
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y[:, 0])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 1e-4

    if not noise:
        # flat non_trainable noise
        likelihood.noise_covar.raw_noise.requires_grad_(False)

    model = GP(train_x, train_y, likelihood)
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with tqdm(range(gp_iter)) as t:
        for _ in t:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            t.set_description(f"GP NLL: {loss.item():.4f}")
            optimizer.step()
    model.eval()
    likelihood.eval()
    return model
