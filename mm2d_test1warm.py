import numpy as np
import ais_lib_icassp2020 as ais
from functools import partial
from scipy.stats import multivariate_normal as mvn
from mpipool import MPIPool
import sys

# Set parameters of the target distribution
d_theta = 2    # dimension
Z_pi = 100     # normalizing constant

# Mixture parameters
rho_pi = np.ones(5)/5
mu_pi = np.array([[-10, -10], [0, 16], [13, 8], [-9, 7], [14, -14]])
sig_pi = np.array([[[2, 0.6], [0.6, 2]], [[2, -0.4], [-0.4, 2]],
                   [[2, 0.8], [0.8, 2]], [[3, 0.], [0., 0.5]], [[2, -0.1], [-0.1, 2]]])

# Create function handle
target = lambda theta: np.log(Z_pi)+np.log(rho_pi[0]*mvn.pdf(theta, mu_pi[0], sig_pi[0])
                                           + rho_pi[1]*mvn.pdf(theta, mu_pi[1], sig_pi[1])
                                           + rho_pi[2]*mvn.pdf(theta, mu_pi[2], sig_pi[2])
                                           + rho_pi[3]*mvn.pdf(theta, mu_pi[3], sig_pi[3])
                                           + rho_pi[4]*mvn.pdf(theta, mu_pi[4], sig_pi[4]))
# Number of Monte Carlo runs
MC_runs = 1000

# Compute true target mean
target_mean_true = np.average(mu_pi, axis=0, weights=rho_pi)

# Sampler parameters
samp_num = 1000
iter_num = 100
mix_num = 25

# Parameters we want to tune
params = np.array([[1, 1], [3, 1], [5, 1], [5, 2], [7, 1], [7, 2], [7, 3], [9, 1], [9, 2], [9, 3], [9, 4]])
L = np.shape(params)[0]

# Variables to store MSE
MAE_store = np.tile(np.zeros((MC_runs, iter_num)), (L, 1, 1))

# Define MPI task
def mpi_worker(var_init, chain_length, chain_thinner, warm_up, i):
    # Run PMC scheme
    result = ais.sgdmh_mpmc(target, d_theta, D=mix_num, M=samp_num, I=iter_num, K=chain_length, var_prop=var_init,
                            bounds=(-20, 20), alpha=2, eta_rho0=0.03, eta_mu0=0.20, eta_prec0=0, g_rho_max=0.05,
                            g_mu_max=1, g_prec_max=0.1, warm_start=warm_up, iter0=250, Kthin=chain_thinner)
    # Compute MSE for estimate in mean
    mae = np.abs(result.evidence-Z_pi)
    return mae


# Create MPI worker pool
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

# Create a list of things you want to iterate over
data_list = [i for i in range(MC_runs)]

# Test a variety of learning rates - Constant
for j in range(L):
    temp_worker = partial(mpi_worker, 1, params[j, 0], params[j, 1], True)
    MAE_store[j] = pool.map(temp_worker, data_list)

# Close pool
pool.close()

# Save variables
np.savez('cmpmc-test1-mm2d-warm_start', parameters=params, perf_mae=MAE_store)
