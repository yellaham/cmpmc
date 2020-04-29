import numpy as np
import other_funcs
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm


# Define a class for adaptive importance samplers with adaptive mixtures
class MPMCSampler:
    def __init__(self, x, log_w, rho, mu, sigma, z_est, mu_est):
        self.particles = x
        self.log_weights = log_w
        self.mix_weights = rho
        self.means = mu
        self.covariances = sigma
        self.evidence = z_est
        self.target_mean = mu_est


# Define a class for adaptive importance samplers which adapt the locations of a population of proposals
class APISSampler:
    def __init__(self, x, log_w, mu, z_est, mu_est):
        self.particles = x
        self.log_weights = log_w
        self.means = mu
        self.evidence = z_est
        self.target_mean = mu_est


# Define a class for Markov Chain Monte Carlo samplers
class MCMCSampler:
    def __init__(self, x):
        self.samples = x


# Implementation of the MH method
def metropolis_hastings(log_target, initial_state, var_x=1, T=1000, burn_in=0, thinning_rate=1):
    """
    Runs the Metropolis-Hastings algorithm using a isotropic Gaussian transition kernel
    :param log_target: Logarithm of the target distribution
    :param initial_state: Initial state in the Markov chain
    :param var_x: Variance of the Markov transition kernel
    :param T: Length of generated Markov chain
    :param burn_in: Number of samples to discard from the beginning of the chain
    :param thinning rate: Thinning of the chain to reduce autocorrelation between samples
    :return particles
    """
    # Determine dimension from intial state
    d = np.shape(initial_state)[0]

    # Initialize storage of particles and log weights
    chain = np.zeros((T+1, d))

    # Initialize the mean of each proposal distribution
    chain[0] = initial_state

    # Loop for the algorithm
    for t in range(T):
        # Propose a particle
        candidate = np.random.multivariate_normal(mean=chain[t], cov=var_x*np.eye(d))

        # Compute the acceptance probability of each sample
        log_pi_num = log_target(candidate)
        log_pi_den = log_target(chain[t])

        # Compute acceptance probability
        alpha = np.minimum(1, np.exp(log_pi_num-log_pi_den))

        # Draw a uniform random number
        u = np.random.rand()

        # Check to see if the sample is accepted
        if u <= alpha:
            chain[t + 1] = candidate
        else:
            chain[t + 1] = chain[t]

    # Apply the burn-in and then thin the chain
    chain = chain[-(T - burn_in):, :]
    chain = chain[0:(T-burn_in):thinning_rate]

    return MCMCSampler(chain)


def cmpmc(log_target, d, D=10, M=50, I=200, K=5, var_prop=1, bounds=(-10, 10), alpha=2, eta_rho0=0,
               eta_mu0=1, eta_prec0=0.1, g_rho_max=0.1, g_mu_max=0.5, g_prec_max=0.25, Kthin=1, var_mcmc=1):
    """
    Runs the proposed controlled mixture population Monte Carlo algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param D: Number of proposals
    :param M: Number of samples to draw
    :param I: Number of iterations
    :param K: Number of MH steps at each iteration for each Markov chain
    :param var_prop: Variance of each proposal distribution
    :param bounds: Prior to generate location parameters over [bounds]**d hypercube
    :param alpha: Renyi divergence parameter (must be greater than 1)
    :param eta_rho0: Initial learning rate for mixture weights
    :param eta_mu0: Initial learning rate for mixand mean
    :param eta_prec0: Initial learning rate for mixand precision matrix
    :param g_rho_max: Maximum norm of mixture weight gradient
    :param g_mu_max: Maximum norm of mixand mean gradient
    :param g_prec_max: Maximum norm of mixand precision matrix gradient
    :param Kthin: Thinning parameter for the MH algorithm
    :param var_mcmc: Pertubration variance of the MCMC state
    :return particles, weights, and estimate of normalizing constant
    """
    # Initialize the weights of the mixture proposal
    rho = np.ones(D) / D

    # Initialize the means of the mixture proposal
    mu = np.random.uniform(bounds[0], bounds[1], (D, d))

    # Initialize the covariances/precisions of the mixture proposal
    sig = np.tile(var_prop * np.eye(d), (D, 1, 1))
    prec = np.tile((1/var_prop) * np.eye(d), (D, 1, 1))

    # Initialize storage of particles and log weights
    particles = np.zeros((M * I, d))
    log_weights = np.ones(M * I) * (-np.inf)
    mix_weights = np.zeros((I + 1, D))
    means = np.zeros((D * (I + 1), d))
    covariances = np.tile(np.zeros((d, d)), (D * (I + 1), 1, 1))

    # Initialize storage of evidence and target mean estimates
    evidence = np.zeros(I)
    target_mean = np.zeros((I, d))

    # Initialize proposal parameters
    mix_weights[0, :] = rho
    means[0:D] = mu
    covariances[0:D] = sig

    # Initialize RMS prop parameters
    vrho = np.zeros(D)
    vmu = np.zeros((D, d))
    vprec = np.zeros((D, d, d))

    # Initialize the states of the D Markov chains
    chain = np.copy(mu)
    # Initialize start counter
    start = 0
    startd = 0

    # Loop for the algorithm
    for i in tqdm(range(I)):
        # Update start counter
        stop = start + M
        stopd = startd + D

        # Generate particles
        idx = np.random.choice(D, M, replace=True, p=rho)
        children = np.random.multivariate_normal(np.zeros(d), np.eye(d), M)
        for j in range(D):
            children[idx == j] = mu[j] + np.matmul(children[idx == j], np.linalg.cholesky(sig[j]).T)
        particles[start:stop] = children

        # Compute log proposal
        log_prop_j = np.zeros((M, D))
        for j in range(D):
            log_prop_j[:, j] = mvn.logpdf(children, mean=mu[j], cov=sig[j], allow_singular=True)
        # Find the maximum log weight
        max_log_weight = np.max(log_prop_j)
        # Determine the equivalent log proposal as
        log_prop = max_log_weight + np.log(np.average(np.exp(log_prop_j-max_log_weight), weights=rho, axis=1))

        # Compute log weights and store
        log_target_eval = log_target(children)
        log_w = log_target_eval - log_prop
        log_weights[start:stop] = log_w

        # Convert log weights to standard weights using LSE and normalize
        w = np.exp(log_w - np.max(log_w))

        # Compute estimate of evidence
        max_log_weight = np.max(log_weights[0:stop])
        weights = np.exp(log_weights[0:stop]-max_log_weight)
        log_z = np.log(1/(M*(i+1)))+max_log_weight+np.log(np.sum(weights))
        evidence[i] = np.exp(log_z)

        # Compute estimate of the target mean
        target_mean[i] = np.average(particles[0:stop, :], axis=0, weights=weights)

        # Copy the parameters of the mixture
        rho_temp = np.copy(rho)
        mu_temp = np.copy(mu)
        sig_temp = np.copy(sig)
        # Adapt the parameters of the proposal distribution
        for j in range(D):
            # PART 1: Obtain target samples by sampling from a Markov chain
            mcmc_run = metropolis_hastings(log_target, chain[j], var_x=var_mcmc, T=K, burn_in=0, thinning_rate=Kthin)
            z_d = mcmc_run.samples

            # PART 2: Evaluate proposal and log proposal
            prop_z = np.zeros(np.shape(z_d)[0])
            prop_z_k = np.zeros((np.shape(z_d)[0], D))
            for jk in range(D):
                prop_z_k[:, jk] = mvn.pdf(z_d, mean=mu_temp[jk], cov=sig_temp[jk], allow_singular=True)
                prop_z += rho_temp[jk] * prop_z_k[:, jk]
            log_prop_z = np.log(prop_z)
            log_target_z = np.asarray([log_target(z_d)]).T

            # PART 3: Compute the gradient of mixand weights, mean, and  precision
            g_rho = 0                           # initialize gradient of mixand weight
            g_mu = np.zeros(d)                  # initialize gradient of mean
            g_prec = np.zeros((d, d))           # initialize gradient of precision matrix
            # Use a loop to make sure you are doing it correctly
            for k in range(np.shape(z_d)[0]):
                factor = np.exp((alpha-1)*(log_target_z[k] - log_prop_z[k]))*(prop_z_k[k, j]/prop_z[k])
                g_rho += factor
                temp_var = np.asmatrix((z_d[k]-mu[j])).T
                g_mu += factor*np.asarray(np.matmul(prec[j], temp_var)).squeeze()
                g_prec += (1/2)*(sig_temp[j] - temp_var*temp_var.T)
            g_rho *= (1-alpha)/np.shape(z_d)[0]
            g_mu *= (rho_temp[j]*(1-alpha))/np.shape(z_d)[0]
            g_prec *= (rho_temp[j]*(1-alpha))/np.shape(z_d)[0]

            # PART 4: Clip the stochastic gradients so they do not get too large
            if np.abs(g_rho) > g_rho_max:
                g_rho = g_rho*(g_rho_max/np.abs(g_rho))
            if np.linalg.norm(g_mu) > g_mu_max:
                g_mu = g_mu*(g_mu_max/np.linalg.norm(g_mu))
            if np.linalg.norm(g_prec) > g_prec_max:
                g_prec = g_prec*(g_prec_max/np.linalg.norm(g_prec))

            # PART 5: Taking the stochastic gradient step
            # Compute square of the gradient
            drho_sq = g_rho ** 2
            dmu_sq = g_mu ** 2
            dprec_sq = g_prec ** 2
            # Update elementwise learning rate parameters
            vrho[j] = 0.9 * vrho[j] + 0.1 * drho_sq
            vmu[j] = 0.9 * vmu[j] + 0.1 * dmu_sq
            vprec[j] = 0.9 * vprec[j] + 0.1 * dprec_sq
            # Compute the learning rates
            eta_rho = eta_rho0 * (np.sqrt(vrho[j]) + 1e-3) ** (-1)
            eta_mu = eta_mu0 * (np.sqrt(vmu[j]) + 1e-3) ** (-1)
            eta_prec = eta_prec0 * (np.sqrt(vprec[j]) + 1e-3) ** (-1)
            # Make stochastic gradient updates
            if eta_rho0 > 0:
                rho[j] = rho[j] - eta_rho * g_rho
            if eta_mu0 > 0:
                mu[j] = mu[j] - eta_mu * g_mu
                chain[j] = mu[j]
            if eta_prec0 > 0:
                # Update and project the precision matrix onto the set of PSD matrices
                prec[j] = prec[j] - eta_prec * g_prec
                # Obtain the corresponding covariance matrix
                sig[j] = np.linalg.inv(prec[j])

        # Add small number to weights and normalize
        rho = other_funcs.projection_simplex_sort(rho + 1e-3)

        # Store parameters
        mix_weights[i + 1] = rho
        means[startd + D: stopd + D] = mu
        covariances[startd + D: stopd + D] = sig

        # Update start counters
        start = stop
        startd = stopd

    # Generate output
    output = MPMCSampler(particles, log_weights, mix_weights, means, covariances, evidence, target_mean)

    return output


def mpmc(log_target, d, D=10, M=50, I=200, var_prop=1, bounds=(-10, 10)):
    """
    Runs the mixture population Monte Carlo algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param D: Number of proposals
    :param M: Number of samples to draw
    :param I: Number of iterations
    :param var_prop: Variance of each proposal distribution
    :param bounds: Prior to generate location parameters over [bounds]**d hypercube
    :return particles, weights, and estimate of normalizing constant
    """
    # Initialize the weights of the mixture proposal
    rho = np.ones(D) / D

    # Initialize the means of the mixture proposal
    mu = np.random.uniform(bounds[0], bounds[1], (D, d))

    # Initialize the covariances of the mixture proposal
    sig = np.tile(var_prop * np.eye(d), (D, 1, 1))

    # Initialize storage of particles and log weights
    particles = np.zeros((M * I, d))
    log_weights = np.ones(M * I) * (-np.inf)
    mix_weights = np.zeros((I + 1, D))
    means = np.zeros((D * (I + 1), d))
    covariances = np.tile(np.zeros((d, d)), (D * (I + 1), 1, 1))

    # Set initial locations to be the parents
    mix_weights[0, :] = rho
    means[0:D] = mu
    covariances[0:D] = sig

    # Initialize storage of evidence and target mean estimates
    evidence = np.zeros(I)
    target_mean = np.zeros((I, d))

    # Initialize start counter
    start = 0
    startd = 0

    # Loop for the algorithm
    for i in tqdm(range(I)):
        # Update start counter
        stop = start + M
        stopd = startd + D

        # Generate particles
        idx = np.random.choice(D, M, replace=True, p=rho)
        children = np.random.multivariate_normal(np.zeros(d), np.eye(d), M)
        for j in range(D):
            children[idx == j] = mu[j] + np.matmul(children[idx == j], np.linalg.cholesky(sig[j]).T)
        particles[start:stop] = children

        # Compute log proposal
        prop = np.zeros(M)
        for j in range(D):
            prop += rho[j] * mvn.pdf(children, mean=mu[j, :], cov=sig[j, :, :], allow_singular=True)
        log_prop = np.log(prop)

        # Compute log weights and store
        log_w = log_target(children) - log_prop
        log_weights[start:stop] = log_w

        # Convert log weights to standard weights using LSE and normalize
        w = np.exp(log_w - np.max(log_w))
        w = w / np.sum(w)

        # Compute estimate of evidence
        max_log_weight = np.max(log_weights[0:stop])
        weights = np.exp(log_weights[0:stop]-max_log_weight)
        log_z = np.log(1/(M*(i+1)))+max_log_weight+np.log(np.sum(weights))
        evidence[i] = np.exp(log_z)

        # Compute estimate of the target mean
        target_mean[i, :] = np.average(particles[0:stop, :], axis=0, weights=weights)

        # Adapt the parameters of the proposal distribution
        for j in range(D):
            # Compute the Rao-Blackwellization factor
            alpha = rho[j] * mvn.pdf(children, mean=mu[j, :], cov=sig[j, :, :]) / prop + 1e-6
            # Update the weight
            rho[j] = np.sum(w * alpha)
            # Compute normalized weights
            wn = w * alpha / rho[j]
            # Update the proposal mean
            mu[j, :] = np.average(children, axis=0, weights=wn)
            # Update the proposal covariance
            sig[j, :, :] = np.cov(children, rowvar=0, bias=True, aweights=wn) + 1e-6 * np.eye(d)

        # Add small number to weights and normalize
        rho = rho / np.sum(rho)

        # Store parameters
        mix_weights[i + 1] = rho
        means[startd + D: stopd + D] = mu
        covariances[startd + D: stopd + D] = sig

        # Update start counters
        start = stop
        startd = stopd

    # Generate output
    output = MPMCSampler(particles, log_weights, mix_weights, means, covariances, evidence, target_mean)

    return output


def apis(log_target, d, D=10, N=5, I=200, var_prop=1, bounds=(-10, 10)):
    """
    Runs the adaptive population importance sampling algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param D: Number of proposals
    :param N: Number of samples to draw per proposal
    :param I: Number of iterations
    :param var_prop: Variance of each proposal distribution
    :param bounds: Prior to generate location parameters over [bounds]**d hypercube
    :return particles, weights, and estimate of normalizing constant
    """
    # Determine the total number of particles
    M = D*N

    # Initialize the means of the mixture proposal
    mu = np.random.uniform(bounds[0], bounds[1], (D, d))

    # Initialize storage of particles and log weights
    particles = np.zeros((M * I, d))
    log_weights = np.ones(M * I) * (-np.inf)
    means = np.zeros((D * (I + 1), d))

    # Set initial locations to be the parents
    means[0:D] = mu

    # Initialize storage of evidence and target mean estimates
    evidence = np.zeros(I)
    target_mean = np.zeros((I, d))

    # Initialize start counter
    start = 0
    startd = 0

    # Loop for the algorithm
    for i in tqdm(range(I)):
        # Update start counter
        stop = start + M
        stopd = startd + D

        # Generate particles
        children = np.repeat(mu, N, axis=0)+np.sqrt(var_prop)*np.random.multivariate_normal(np.zeros(d), np.eye(d), M)
        particles[start:stop] = children

        # Compute log proposal
        log_prop_j = np.zeros((M, D))
        prop = np.zeros(M)
        for j in range(D):
            log_prop_j[:, j] = mvn.pdf(children, mean=mu[j], cov=var_prop*np.eye(d), allow_singular=True)
            prop += log_prop_j[:, j]/D
        log_prop = np.log(prop)

        # Compute log weights and store
        log_target_eval = log_target(children)
        log_w = log_target_eval - log_prop
        log_weights[start:stop] = log_w

        # Compute estimate of evidence
        max_log_weight = np.max(log_weights[0:stop])
        weights = np.exp(log_weights[0:stop]-max_log_weight)
        log_z = np.log(1/(M*(i+1)))+max_log_weight+np.log(np.sum(weights))
        evidence[i] = np.exp(log_z)

        # Compute estimate of the target mean
        target_mean[i, :] = np.average(particles[0:stop, :], axis=0, weights=weights)

        # Adapt the parameters of the proposal distribution
        start_j = 0
        for j in range(D):
            # Update stop parameter
            stop_j = start_j + N
            # Get local children
            children_j = children[start_j:stop_j]
            # Obtain local log weights
            log_wj = log_target_eval[start_j:stop_j] - log_prop_j[start_j:stop_j, j]
            # Convert to weights using LSE
            wj = np.exp(log_wj - np.max(log_wj))
            # Normalize the weights
            wjn = wj/np.sum(wj)
            # Update the proposal mean
            mu[j] = np.average(children_j, axis=0, weights=wjn)
            # Update start parameter
            start_j = stop_j

        # Store parameters
        means[startd + D: stopd + D] = mu

        # Update start counters
        start = stop
        startd = stopd

    # Generate output
    output = APISSampler(particles, log_weights, means, evidence, target_mean)

    return output


def pimais(log_target, d, D=10, N=5, I=200, var_prop=1, bounds=(-10, 10), K=1):
    """
    Runs the population Monte Carlo algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param D: Number of proposals
    :param N: Number of samples to draw per proposal
    :param I: Number of iterations
    :param K: Number of MCMC steps
    :param var_prop: Variance of each proposal distribution
    :param bounds: Prior to generate location parameters over [bounds]**d hypercube
    :return particles, weights, and estimate of normalizing constant
    """
    # Determine the total number of particles
    M = D*N

    # Initialize the means of the mixture proposal
    mu = np.random.uniform(bounds[0], bounds[1], (D, d))

    # Initialize storage of particles and log weights
    particles = np.zeros((M * I, d))
    log_weights = np.ones(M * I) * (-np.inf)
    means = np.zeros((D * (I + 1), d))

    # Set initial locations to be the parents
    means[0:D] = mu

    # Initialize storage of evidence and target mean estimates
    evidence = np.zeros(I)
    target_mean = np.zeros((I, d))

    # Initialize the states of the D Markov chains
    chain = mu

    # Initialize start counter
    start = 0
    startd = 0

    # Loop for the algorithm
    for i in tqdm(range(I)):
        # Update start counter
        stop = start + M
        stopd = startd + D

        # Generate particles
        children = np.repeat(chain, N, axis=0)+np.sqrt(var_prop)*np.random.multivariate_normal(np.zeros(d),
                                                                                               np.eye(d), M)
        particles[start:stop] = children

        # Compute log proposal
        log_prop_j = np.zeros((M, D))
        prop = np.zeros(M)
        for j in range(D):
            log_prop_j[:, j] = mvn.pdf(children, mean=mu[j], cov=var_prop*np.eye(d), allow_singular=True)
            prop += log_prop_j[:, j]/D
        log_prop = np.log(prop)

        # Compute log weights and store
        log_target_eval = log_target(children)
        log_w = log_target_eval - log_prop
        log_weights[start:stop] = log_w

        # Compute estimate of evidence
        max_log_weight = np.max(log_weights[0:stop])
        weights = np.exp(log_weights[0:stop]-max_log_weight)
        log_z = np.log(1/(M*(i+1)))+max_log_weight+np.log(np.sum(weights))
        evidence[i] = np.exp(log_z)

        # Compute estimate of the target mean
        target_mean[i, :] = np.average(particles[0:stop, :], axis=0, weights=weights)

        # Adapt the parameters of the proposal distribution
        for j in range(D):
            # PART 1: Obtain target samples by sampling from a Markov chain
            z_old = chain[j]
            for k in range(K):
                # Propagation using Markov transition kernel
                z_star = np.random.multivariate_normal(z_old, np.eye(d))
                # Compute the acceptance probability (symmetric transition kernel)
                ap = np.exp(log_target(z_star) - log_target(z_old))
                # Check to see if the sample should be accepted
                if np.random.rand() < ap:
                    chain[j] = z_star
                    z_old = z_star

        # Store parameters
        means[startd + D: stopd + D] = chain

        # Update start counters
        start = stop
        startd = stopd

    # Generate output
    output = APISSampler(particles, log_weights, means, evidence, target_mean)

    return output



