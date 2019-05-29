# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def generate_sample(n, alpha):
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1
    mean1, mean2 = np.array([2, 0]), np.array([-2, 0])
    cov = np.array([[1, 0], [0, 9]])
    x1 = np.random.multivariate_normal(mean1, cov, n1).transpose()
    x2 = np.random.multivariate_normal(mean2, cov, n2).transpose()
    return x1, x2


def sampling_normal(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)


def main():
    np.random.seed(0)

    # settings
    n = 100
    alpha = 0.3

    mu_1 = np.array([2, 0])
    mu_2 = np.array([-2, 0])
    sigma = np.array([[1, 0], [0, 9]])


    # variables
    n_1 = sum(np.random.rand(n) < alpha)
    n_2 = n - n_1

    p_1 = alpha
    p_2 = 1 - alpha


    # generate data
    x_1 = sampling_normal(mu_1, sigma, n_1)
    x_2 = sampling_normal(mu_2, sigma, n_2)
    # print(x_1.shape)


    # calc
    constant = 0.0
    sigma_inv = np.linalg.inv(sigma)


    # log probs
    def log_probabilities(x):
        logp_1x = mu_1.T.dot(sigma_inv).dot(x.T) - (1/2) * mu_1.T.dot(sigma_inv).dot(mu_1) + np.log(p_1) + constant
        logp_2x = mu_2.T.dot(sigma_inv).dot(x.T) - (1/2) * mu_2.T.dot(sigma_inv).dot(mu_2) + np.log(p_2) + constant
        return logp_1x, logp_2x

    x = x_1
    logp_1x, logp_2x = log_probabilities(x)
    is_1 = (logp_1x > logp_2x)
    print('1: #Data: {}\t#Correct: {}\tAcc: {:.3f}'.format(len(x), is_1.sum(), is_1.sum()/len(x)))

    x = x_2
    logp_1x, logp_2x = log_probabilities(x)
    is_2 = (logp_1x < logp_2x)
    print('2: #Data: {}\t#Correct: {}\tAcc: {:.3f}'.format(len(x), is_2.sum(), is_2.sum()/len(x)))


    # coeffs of decision boundary
    a = (mu_1.T - mu_2.T).dot(sigma_inv).T
    b = -(1/2) * ((mu_1.T).dot(sigma_inv).dot(mu_1) - (mu_2.T).dot(sigma_inv).dot(mu_2)) + np.log(p_1/p_2)
    # _x = np.arange(-5, 5, 0.1)

    plt.title(r'$\alpha = {}$'.format(alpha))
    plt.scatter(x_1[:, 0], x_1[:, 1], marker='o')
    plt.scatter(x_2[:, 0], x_2[:, 1], marker='x')
    #plt.plot(-a[0]/a[1]*_x + b/a[1], _x)
    plt.show()


if __name__ == '__main__':
    main()
