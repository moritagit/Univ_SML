# -*- coding: utf-8 -*-


import pathlib
import numpy as np
import matplotlib.pyplot as plt


def fisher(x, mean, cov_inv, p_y):
    logp = mean.T.dot(cov_inv).dot(x.T) - (1/2)*mean.T.dot(cov_inv).dot(mean) + np.log(p_y)
    return logp


def mahalanobis(x, mean, cov, p_y, eps=1e-6):
    cov_inv = np.linalg.inv(cov + eps*np.eye(len(cov)))
    logp = -(1/2)*np.diag((x - mean.T).dot(cov_inv).dot((x - mean.T).T))
    logp += - (1/2)*np.log(np.linalg.det(cov)) + np.log(p_y)
    return logp


def main():
    np.random.seed(0)

    datadir = pathlib.Path().cwd().parent / 'data'

    n_category = 10
    categories = list(range(10))

    # train
    data = []
    means = []
    covs = []
    for category in categories:
        data_path = datadir / 'digit_train{}.csv'.format(category)
        _data = np.loadtxt(str(data_path), delimiter=',')
        mean = np.mean(_data, axis=0)
        cov = np.cov(_data.T)
        data.append(_data)
        means.append(mean)
        covs.append(cov)
    cov_train = np.zeros_like(covs[0])
    for i in range(n_category):
        cov_train += covs[i]
    cov_train /= n_category
    cov_train_inv = np.linalg.inv(cov_train + 1e-8*np.eye(len(cov_train)))


    # test
    n_test = 0
    data_test = []
    for category in categories:
        data_path = datadir / 'digit_test{}.csv'.format(category)
        _data = np.loadtxt(str(data_path), delimiter=',')
        n_test += len(_data)
        data_test.append(_data)

    confusion_matrix = np.zeros((n_category, n_category))
    for y, data in enumerate(data_test):
        print('Category: {}\t'.format(y), end='')
        n_data = len(data)
        p_y = n_data / n_test
        preds = []
        for category in categories:
            mean = means[category]
            cov = covs[category]
            logp = fisher(data, mean, cov_train_inv, p_y)
            # logp = mahalanobis(data, mean, cov, p_y)
            preds.append(logp)
        preds = np.array(preds).T
        flag = np.argmax(preds, axis=1)
        for category in categories:
            n = (flag == category).sum()
            confusion_matrix[y, category] = n
        n_correct = (flag == y).sum()
        acc = n_correct / n_data
        print('#Data: {}\t#Crr: {}\tAcc: {:.3f}'.format(n_data, n_correct, acc))

    print()
    print('Confusion Matrix\n', confusion_matrix)
    print()

    n_crr_all = np.diag(confusion_matrix).sum()
    n_data_all = 200 * 10
    acc_all = n_crr_all / n_data_all
    print('All\t#Data: {}\t#Crr: {}\tAcc: {:.3f}'.format(n_data_all, n_crr_all, acc_all))


if __name__ == '__main__':
    main()
