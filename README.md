# Controlled Mixture Population Monte Carlo (CMPMC)
 This repository contains a Python implementation of the population Monte Carlo algorithm proposed in the following 
 paper:
 
 [Enhanced mixture population Monte Carlo via stochastic optimization and Markov chain Monte Carlo sampling](https://ieeexplore.ieee.org/document/9053410)
 
Here is a description of each file:
1. main.py - Source code for running the algorithm (along with other implemented samplers) on a toy example where the 
target distribution is a mixture of Gaussians. The code runs independent Monte Carlo simulations of the sampler, where
multiprocessing package is used for parallelization. One can use this base code to replicate the results reported in 
the above paper.
2. samplers.py - Library containing the implementation of the proposed sampler, as well as the samplers that it is
compared to in the paper, i.e., controlled mixture population Monte Carlo (CMPMC), mixture population Monte Carlo 
(MPMC), adaptive population importance sampling (APIS), and parallel interacting Markov adaptive importance
sampling (PIMAIS). 
3. other_funcs.py - Code for plotting a two-dimensional contour, evaluating a Gaussian mixture distribution, and for 
projecting onto the probability simplex. The code for projecting onto the probability simplex can be found in
[this repo](https://github.com/michael-lash/Prophit).

