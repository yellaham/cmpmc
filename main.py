# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import samplers as ais
import multiprocessing
from functools import partial
from scipy.stats import multivariate_normal as mvn

# Dimension of the target
d_theta = 2

# Set parameters of the target distribution
Z_pi = 100     # normalizing constant
rho_pi = np.ones(5)/5   # mixture weights
# Mean of each mixand
mu_pi = np.array([[-10, -10], [0, 16], [13, 8], [-9, 7], [14, -14]])
# Covariance matrix of each mixand
sig_pi = np.array([[[2, 0.6], [0.6, 2]], [[2, -0.4], [-0.4, 2]],
                   [[2, 0.8], [0.8, 2]], [[3, 0.], [0., 0.5]], [[2, -0.1], [-0.1, 2]]])

# Create a lambda which is the target probability distributionv (2D, 5 mode target)
log_target = lambda theta: np.log(Z_pi) + np.log(rho_pi[0] * mvn.pdf(theta, mu_pi[0], sig_pi[0])
                                                 + rho_pi[1] * mvn.pdf(theta, mu_pi[1], sig_pi[1])
                                                 + rho_pi[2] * mvn.pdf(theta, mu_pi[2], sig_pi[2])
                                                 + rho_pi[3] * mvn.pdf(theta, mu_pi[3], sig_pi[3])
                                                 + rho_pi[4] * mvn.pdf(theta, mu_pi[4], sig_pi[4]))

# Compute true target mean
target_mean_true = np.average(mu_pi, axis=0, weights=rho_pi)


# Sampler parameters
samp_num = 200
iter_num = 500
mix_num = 25
samp_per_mix = int(np.floor(samp_num/mix_num))
var0 = 1


# Define the input process
def worker(var_init, chain_length, chain_thinner, i):
    np.random.seed()
    # Run the proposed CMPMC algorithm
    result = ais.cmpmc(log_target, d_theta, D=mix_num, M=samp_num, I=iter_num, K=chain_length, var_prop=var_init,
                       bounds=(-20, 20), alpha=2, eta_rho0=0, eta_mu0=0.9, eta_prec0=1e-2,
                       g_rho_max=0.01, g_mu_max=0.5, g_prec_max=0.1, Kthin=chain_thinner)
    # Compute MSE for estimate in mean
    mse = (1 / d_theta) * np.sum((result.target_mean - target_mean_true) ** 2, axis=1)
    mae = np.abs(result.evidence-Z_pi)
    print("END SIMULATION %d" %i)
    return mae


# Define the number of MC runs
MC_runs = 500

# Learning rates to test
params = np.array([[1, 1], [3, 1], [5, 1], [5, 2], [7, 1], [7, 2], [9, 1], [9, 2], [9, 3]])
L = np.shape(params)[0]

# Set up matrices for different optimizers
MAE_store = np.tile(np.zeros((MC_runs, iter_num)), (L, 1, 1))

# Create a list of things you want to iterate over
data_list = [i for i in range(MC_runs)]

if __name__ == '__main__':
    # Number of cores for parallel processing
    num_cores = multiprocessing.cpu_count()
    print("The total number of CPUs is", num_cores)

    # Test a variety of learning rates
    plt.plot()
    for j in range(L):
        temp_worker = partial(worker, var0, params[j, 0], params[j, 1])
        pool = multiprocessing.Pool(processes=num_cores)
        output = pool.imap_unordered(temp_worker, data_list)
        pool.close()
        MAE_store[j] = np.array(list(output))
        print("###################")
        print("FINISHED POOL %d" % j)
        print("###################")
        plt.semilogy(np.median(MAE_store[j], axis=0), label='K = %d, xi= %d' % (params[j, 0], params[j, 1]))

    plt.title('Performance')
    plt.xlabel('Iteration')
    plt.ylabel('Median MSE')
    plt.legend()
    plt.show()

