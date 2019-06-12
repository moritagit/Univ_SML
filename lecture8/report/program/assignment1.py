# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


def beta_distribution(pi, a, b):
    value = (
        (math.gamma(a + b) / (math.gamma(a) * math.gamma(b)))
        * pi**(a-1) * (1 - pi)**(b-1)
        )
    return value


def make_beta(a, b):
    def _beta(pi):
        return beta_distribution(pi, a, b)
    return _beta


def integrate(distribution, lower=0.0, upper=1.0, dx=0.01,):
    x = np.arange(lower, upper+dx, dx)
    dist = distribution(x)
    p = (dist*dx).sum()
    return p


def main():
    # settings
    n_pos, n_neg = 4, 1  # data
    ab_cands = [(1, 1), (0.1, 0.1), (5, 5)]  # prior dist
    thresholds = [0.5, 0.8]
    dx = 0.0001
    fig_path = '../figures/assignment1_result.png'

    # calc
    x_axis = np.arange(0, 1.0, 0.01)
    n_row = 1
    n_col = len(ab_cands)
    fig = plt.figure(figsize=(5*n_col, 5*n_row))
    for i, (a, b) in enumerate(ab_cands):
        distribution = make_beta(a+n_pos, b+n_neg)
        for threshold in thresholds:
            p = integrate(distribution, lower=threshold, dx=dx)
            print(f'a = {a}  b = {b}  p(pi > {threshold}) = {p}')

        # plot
        dist = distribution(x_axis)
        label = f'a = {a}  b = {b}'
        ax = fig.add_subplot(n_row, n_col, i+1)
        ax.set_title(label)
        ax.set_xlim(0, 1)
        for threshold in thresholds:
            ax.vlines(threshold, 0, dist.max()+1, linestyle='dashed')
        ax.plot(x_axis, dist, label='posterior')
        ax.plot(x_axis, beta_distribution(x_axis, a, b), label='prior')
        ax.legend()
    if fig_path:
        plt.savefig(str(fig_path))
    plt.show()



if __name__ == '__main__':
    main()
