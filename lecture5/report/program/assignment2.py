# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def generate_data(n=1000):
    x = np.concatenate([
        np.random.rand(n, 1),
        np.random.randn(n, 1)
        ], axis=1)
    x[0, 1] = 6   # outlier

    # Standardization
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    M = np.array([[1, 3], [5, 3]])
    x = x.dot(M.T)
    x = np.linalg.inv(sqrtm(np.cov(x, rowvar=False))).dot(x.T).T
    return x


def metric_s4(s, derivative=0):
    if derivative == 0:
        met = s**4
    elif derivative == 1:
        met = 4*s**3
    elif derivative == 2:
        met = 12*s**2
    else:
        raise ValueError('Derivatives more than second are not defined. But the input was: {}'.format(derivative))
    return met


def metric_logcosh(s, derivative=0):
    if derivative == 0:
        met = np.log(np.cosh(s))
    elif derivative == 1:
        met = np.tanh(s)
    elif derivative == 2:
        met = 1 - np.tanh(s)**2
    else:
        raise ValueError('Derivatives more than second are not defined. But the input was: {}'.format(derivative))
    return met


def metric_exp(s, derivative=0):
    if derivative == 0:
        met = - np.exp(-(s**2)/2)
    elif derivative == 1:
        met = s*np.exp(-(s**2)/2)
    elif derivative == 2:
        met = (1 - s**2) * np.exp(-(s**2)/2)
    else:
        raise ValueError('Derivatives more than second are not defined. But the input was: {}'.format(derivative))
    return met


def normalize(b):
    b = b / np.linalg.norm(b)
    if b[0] < 0:
        b *= -1
    return b


def update(b, x_whitened, metric):
    n = len(x_whitened)
    s = x_whitened.dot(b)
    b_new = (
        (np.mean(metric(s, 2))) * b
        - (1/n) * np.sum(x_whitened * metric(s, 1)[:, np.newaxis], axis=0)
        )
    return b_new


def train(x_whitened, metric, max_iter=100, eps=1e-4, n=5):
    d = x_whitened.shape[1]

    # initialize b
    b = np.random.randn(d)
    b = normalize(b)

    b_olds = []
    for i in range(max_iter):
        b_old = b.copy()
        b = update(b, x_whitened, metric)
        b = normalize(b)

        # if converge, break
        if len(b_olds) < n:
            b_olds.append(b_old)
        else:
            b_olds[:-1] = b_olds[1:]
            b_olds[-1] = b_old
            b_olds = np.array(b_olds)
            diffs = np.sqrt(np.sum((b_olds - b)**2, axis=1))
            if diffs.max() < eps:
                break
    n_iter = i + 1
    return b, n_iter


def main():
    # settnigs
    n_sample = 1000
    metric, metric_name = metric_s4, 's4'
    #metric, metric_name = metric_logcosh, 'logcosh'
    #metric, metric_name = metric_exp, 'exp'

    offset = 1.0
    np.random.seed(0)
    scatter_path = f'../figures/assignment2_result_{metric_name}_n{n_sample}_scatter.png'
    hist_path = f'../figures/assignment2_result_{metric_name}_n{n_sample}_hist.png'


    # load data
    x = generate_data(n_sample)
    #x_whitened = (x - np.mean(x, axis=0)) / np.std(x, axis=0)


    # train
    b, n_iter = train(x, metric, max_iter=1000, eps=1e-4, n=5)


    # result
    print(f'Metric: {metric_name}')
    print(f'#Data: {n_sample} \t #Iter: {n_iter}')
    print('b: {} (norm = {})'.format(b, np.linalg.norm(b)))


    # plot scatter
    scale = 1e3
    x0_max, x0_min = x[:, 0].max(), x[:, 0].min()
    x1_max, x1_min = x[:, 1].max(), x[:, 1].min()


    plt.scatter(x[:, 0], x[:, 1], color='royalblue', s=8)
    plt.quiver(
        0, 0, b[0]*scale, b[1]*scale,
        color='darkcyan', angles='xy', scale_units='xy', scale=6.5,
        )
    plt.quiver(
        0, 0, -b[0]*scale, -b[1]*scale,
        color='darkcyan', angles='xy', scale_units='xy', scale=6.5,
        )
    plt.xlim([x0_min-offset, x0_max+offset])
    plt.ylim([x1_min-offset, x1_max+offset])
    plt.savefig(scatter_path)
    plt.show()


    # plot hist
    x_casted = x.dot(b)
    plt.hist(x_casted, bins=50, rwidth=0.9)
    plt.savefig(hist_path)
    plt.show()


if __name__ == '__main__':
    main()
