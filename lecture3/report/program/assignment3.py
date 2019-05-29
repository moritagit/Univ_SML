# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def generate_sample(n, theta):
    sample = (np.random.rand(n) < theta)
    return sample


def main():
    # settings
    theta = 0.3
    n_sample = 100000
    n_mle = 1000
    np.random.seed(0)


    theta_mle_list = []
    fisher_matrices = []
    for _ in range(n_mle):
        sample = generate_sample(n_sample, theta)
        n_o = sample.sum()
        theta_mle = n_o / n_sample
        fisher_matrix = (1/n_sample) * ((n_sample/(theta*(1-theta))) * (theta_mle - theta))**2
        theta_mle_list.append(theta_mle)
        fisher_matrices.append(fisher_matrix)

    mean = np.mean(theta_mle_list)
    cov = np.cov(theta_mle_list)
    F = np.mean(fisher_matrices)

    W, p = stats.shapiro(theta_mle_list)
    is_normal_dist = (p > 0.05)

    print('n: {} \t n_MLE: {}'.format(n_sample, n_mle))
    print('True: theta*: {} \t 1/nF: {} \t F: {}'.format(theta, 1/(n_sample*F), F))
    print('MLE: mean: {:.4f} \t cov: {}'.format(mean, cov))
    print('is normal: {} \t W: {} \t p: {}'.format(is_normal_dist, W, p))


    # histgram
    plt.hist(theta_mle_list, bins=50)
    plt.xlabel(r'$\theta_\mathrm{MLE}$')
    plt.ylabel('freq')
    plt.savefig('../figures/hist_n{}.png'.format(n_sample))
    plt.show()

    # QQ plot
    stats.probplot(theta_mle_list, dist='norm', plot=plt)
    plt.xlabel('Quantailes')
    plt.ylabel('')
    plt.savefig('../figures/qqplot_n{}.png'.format(n_sample))
    plt.show()


if __name__ == '__main__':
    main()
