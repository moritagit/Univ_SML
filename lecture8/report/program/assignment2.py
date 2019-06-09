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


def expected_number(posterior_dist, l):
    def _num(pi):
        return l * ((1 - pi)/pi) * posterior_dist(pi)
    return _num


def integrate(distribution, lower=0.0, upper=1.0, dx=0.01,):
    x = np.arange(lower, upper+dx, dx)
    dist = distribution(x)
    p = (dist*dx).sum()
    return p


def main():
    # settings
    k, m = 5, 15  # data
    l = 2
    a, b = 1, 1   # prior dist
    threshold = 0.1
    dx = 0.0001

    # calc
    posterior_dist = make_beta(a+k, b+m)
    n_neg_expected = integrate(expected_number(posterior_dist, l), dx=dx, lower=dx)
    print(f'a = {a}  b = {b}  E[neg] = {n_neg_expected:.2f}  #Expected = {l+n_neg_expected:.2f}')

    p = integrate(posterior_dist, upper=threshold, dx=dx)
    print(f'a = {a}  b = {b}  p(pi < {threshold}) = {p:.4f}')


if __name__ == '__main__':
    main()
