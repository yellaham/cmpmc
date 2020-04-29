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
target = lambda theta: np.log(Z_pi) + np.log(rho_pi[0]*mvn.pdf(theta, mu_pi[0], sig_pi[0])
                                             + rho_pi[1]*mvn.pdf(theta, mu_pi[1], sig_pi[1])
                                             + rho_pi[2]*mvn.pdf(theta, mu_pi[2], sig_pi[2])
                                             + rho_pi[3]*mvn.pdf(theta, mu_pi[3], sig_pi[3])
                                             + rho_pi[4]*mvn.pdf(theta, mu_pi[4], sig_pi[4]))
# Number of Monte Carlo runs
MC_runs = 1000

# Compute true target mean
target_mean_true = np.average(mu_pi, axis=0, weights=rho_pi)

# Sampler parameters
samp_num = 200
iter_num = 500
mix_num = 25
samp_per_mix = int(np.floor(samp_num/mix_num))
var0 = 1

# Parameters we want to tune for PIMAIS method
params = np.arange(1, 5, 1)
L = np.shape(params)[0]

# Variables to store MSE
MAE_store_PIMAIS = np.tile(np.zeros((MC_runs, iter_num)), (L, 1, 1))

# Define MPI task
def mpi_worker(method, var_init, chain_length, i):
    # Set random seed
    np.random.seed()
    # Run APIS scheme
    if method == 0:
        result = ais.apis(target, d_theta, D=mix_num, N=samp_per_mix, I=iter_num, var_prop=var_init, bounds=(-20, 20))
    # Run MPMC scheme
    elif method == 1:
        result = ais.mpmc(target, d_theta, D=mix_num, M=samp_num, I=iter_num, var_prop=var_init, bounds=(-20, 20))
    # Run PIMAIS scheme
    elif method == 2:
        result = ais.pimais(target, d_theta, D=mix_num, N=samp_per_mix, I=iter_num, K=chain_length, var_prop=var_init,
                            bounds=(-20, 20))
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

# Run APIS method
temp_worker = partial(mpi_worker, 0, var0, 0)
MAE_store_APIS = pool.map(temp_worker, data_list)

# Run APIS method
temp_worker = partial(mpi_worker, 1, var0, 0)
MAE_store_MPMC = pool.map(temp_worker, data_list)

# Test a variety of chain lengths for PI-MAIS
for j in range(L):
    temp_worker = partial(mpi_worker, 2, var0, params[j])
    MAE_store_PIMAIS[j] = pool.map(temp_worker, data_list)

# Close pool
pool.close()

# Print
print("Changed Script")

# Save variables
np.savez('cmpmc-test2-mm2d', chain_lengths=params, perf_mae_apis=MAE_store_APIS, perf_mae_mpmc=MAE_store_MPMC,
         perf_mae_pimais=MAE_store_PIMAIS)
