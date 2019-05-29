# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=3000):
    x = np.zeros(n)
    u = np.random.rand(n)
    index1 = np.where((0 <= u) & (u < 1 / 8))
    x[index1] = np.sqrt(8 * u[index1])
    index2 = np.where((1 / 8 <= u) & (u < 1 / 4))
    x[index2] = 2 - np.sqrt(2 - 8 * u[index2])
    index3 = np.where((1 / 4 <= u) & (u < 1 / 2))
    x[index3] = 1 + 4 * u[index3]
    index4 = np.where((1 / 2 <= u) & (u < 3 / 4))
    x[index4] = 3 + np.sqrt(4 * u[index4] - 2)
    index5 = np.where((3 / 4 <= u) & (u <= 1))
    x[index5] = 5 - np.sqrt(4 - 4 * u[index5])
    return x


def split(x, n, shuffle=True):
    n_data = len(x)
    n_data_split = n_data // n
    n_abandoned = n_data % n
    if n_abandoned != 0:
        print(f'Warning: {n_abandoned} samples are abandoned')
    if shuffle:
        x_split = np.random.permutation(x)
    else:
        x_split = x.copy()
    x_split = [x_split[i:i+n_data_split] for i in range(0, n_data, n_data_split)]
    return x_split


def make_train_data(x, n_split, i):
    x_valid = x[i]
    x_train = []
    for j in range(n_split):
        if j != i:
            x_train.extend(x[j])
    x_train = np.array(x_train)
    return x_train, x_valid


def gauss_kernel(x, d):
    k = (1/(2*np.pi)**(d/2)) * np.exp(-(1/2)*x**2)
    #print((1/(2*np.pi)**(d/2)), k, -(1/2)*x.T.dot(x))
    return k


def kernel_density(x, h, x_axis, kernel):
    n = x.shape[0]
    if len(x.shape) == 1:
        d = 1
    else:
        d = x.shape[1]
    prob = np.zeros(len(x_axis))
    for x_i in x:
        kernel_input = (x_axis - x_i) / h
        #print(kernel_input)
        prob += kernel(kernel_input, d)
    prob = prob / (n * h**d)
    #print(x)
    #print(prob)
    return prob


def estimate_kernel_density(
        x, n_split, bandwidth_list, kernel,
        offset=1.0, num=100,
        path=None,
        ):
    x_axis = np.linspace(x.min(), x.max(), num)
    x_split = split(x, n=n_split, shuffle=True)

    n_bandwidth = len(bandwidth_list)
    n_row = 1
    n_col = n_bandwidth
    fig = plt.figure(figsize=(n_col*4, n_row*6))

    lcv_list = []
    for i, bandwidth in enumerate(bandwidth_list):
        # calc LCV by likelihood cross validation
        lcv_list_tmp = []
        for j in range(n_split):
            x_train, x_valid = make_train_data(x_split, n_split, j)
            p = kernel_density(x_valid, bandwidth, x_train, kernel)
            #lcv = p.sum()
            lcv = np.log(p).sum()
            lcv_list_tmp.append(lcv)
        lcv = np.mean(lcv_list_tmp)
        lcv_list.append(lcv)

        # plot
        p = kernel_density(x_train, bandwidth, x_axis, kernel)
        ax = fig.add_subplot(n_row, n_col, (i+1))
        ax.set_title(f'$h = {bandwidth}$')
        ax.hist(x, bins=50, normed=True)
        ax.plot(x_axis, p)
        ax.set_ylim([0, 1.0])

    if path:
        plt.savefig(str(path))
    plt.show()
    return lcv_list


def main():
    # settings
    n_sample = 3000
    n_split = 10
    h_list = [0.01, 0.05, 0.1, 0.5,]  # global
    #h_list = [0.05, 0.075, 0.1, 0.15]  # local
    offset = 1.0
    num = 100
    fig_path = '../figures/assignment1_result_global.png'
    np.random.seed(0)


    # load data
    x = generate_data(n_sample)
    #print('x shape: {}'.format(x.shape))
    #plt.hist(x, bins=50)
    #plt.show()


    # train
    lcv_list = estimate_kernel_density(
        x, n_split, h_list, gauss_kernel,
        offset=offset, num=num,
        path=fig_path,
        )

    # result
    form = '{:5.2f}'
    tab = '  '
    string_h = '{:3}'.format('h')
    string_lcv = '{:3}'.format('LCV')
    for h, lcv in zip(h_list, lcv_list):
        string_h += tab + form.format(h)
        string_lcv += tab + form.format(lcv)
    result = string_h + '\n' + string_lcv
    print(result)


if __name__ == '__main__':
    main()
