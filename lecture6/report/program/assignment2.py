# -*- coding: utf-8 -*-


import pathlib
import numpy as np
import matplotlib.pyplot as plt


def load_data(n_label=None, n_train=None, n_test=None):
    data_dir = '../data/'
    data_dir = pathlib.Path(data_dir)
    categories = list(range(10))
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for category in categories[:n_label]:
        # train data
        data_path = data_dir / 'digit_train{}.csv'.format(category)
        data = np.loadtxt(str(data_path), delimiter=',')[:n_train]
        n_data = len(data)
        train_X.extend(data)
        train_y.extend(np.ones(n_data) * category)

        # test data
        data_path = data_dir / 'digit_test{}.csv'.format(category)
        data = np.loadtxt(str(data_path), delimiter=',')[:n_test]
        n_data = len(data)
        test_X.extend(data)
        test_y.extend(np.ones(n_data) * category)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    labels = categories[:n_label]
    return train_X, train_y, test_X, test_y, labels


def shuffle(data_X, data_y):
    n_data = len(data_y)
    indices = np.arange(n_data)
    np.random.shuffle(indices)
    data_X_shuffled = data_X[indices]
    data_y_shuffled = data_y[indices]
    return data_X_shuffled, data_y_shuffled


def split(data_X, data_y, n):
    n_data = len(data_y)
    n_data_split = n_data // n
    n_abandoned = n_data % n
    if n_abandoned != 0:
        print(f'Warning: {n_abandoned} samples are abandoned')
    data_X_split = [data_X[i:i+n_data_split] for i in range(0, n_data, n_data_split)]
    data_y_split = [data_y[i:i+n_data_split] for i in range(0, n_data, n_data_split)]
    return data_X_split, data_y_split


def make_train_data(train_X, train_y, n_split, i):
    train_X_valid = train_X[i]
    train_y_valid = train_y[i]
    train_X_train = []
    train_y_train = []
    for j in range(n_split):
        if j != i:
            train_X_train.extend(train_X[j])
            train_y_train.extend(train_y[j])
    train_X_train = np.array(train_X_train)
    train_y_train = np.array(train_y_train)
    return train_X_train, train_y_train, train_X_valid, train_y_valid


def knn(train_X, train_y, test_X, k_list, save_memory=False):
    if save_memory:
        n_train = train_X.shape[0]
        n_test = test_X.shape[0]
        dist_matrix = np.zeros((n_test, n_train))
        for i in range(n_test):
            test_X_i = test_X[i]
            dist_matrix[i, :] = np.sum((train_X - test_X_i[np.newaxis, :])**2, axis=1)
    else:
        dist_matrix = np.sqrt(
            np.sum((train_X[None] - test_X[:, None])**2, axis=2)
            )

    sorted_index_matrix = np.argsort(dist_matrix, axis=1)
    ret_matrix = None
    for k in k_list:
        knn_label = train_y[sorted_index_matrix[:, :k]]
        label_sum_matrix = None
        for i in range(10):
            predict = np.sum(np.where(knn_label == i, 1, 0), axis=1)[:, None]
            if label_sum_matrix is None:
                label_sum_matrix = predict
            else:
                label_sum_matrix = np.concatenate(
                    [label_sum_matrix, predict],
                    axis=1)
        if ret_matrix is None:
            ret_matrix = np.argmax(label_sum_matrix, axis=1)[:, None]
        else:
            ret_matrix = np.concatenate([
                ret_matrix,
                np.argmax(label_sum_matrix, axis=1)[:, None]
                ], axis=1)
    #asert ret_matrix.shape == (len(test_x), len(k_list))
    return ret_matrix


def train(train_X, train_y, k_list, save_memory=False):
    n_split = len(train_y)
    n_corrects_list = []
    for i in range(n_split):
        train_X_train, train_y_train, train_X_valid, train_y_valid = make_train_data(
            train_X, train_y, n_split, i
            )
        y_preds = knn(
            train_X_train, train_y_train, train_X_valid,
            k_list, save_memory=save_memory
            )
        result = (y_preds == train_y_valid[:, np.newaxis])
        n_corrects = result.astype(int).sum(axis=0)
        n_corrects_list.append(n_corrects)
    n_corrects_list = np.array(n_corrects_list)
    n_corrects = n_corrects_list.mean(axis=0)
    return n_corrects


def test(train_X, train_y, test_X, test_y, k, labels):
    n_label = len(labels)
    confusion_matrix = np.zeros((n_label, n_label), dtype=int)
    n_data_all = len(test_y)
    result = {}
    print('Test')

    preds_all = knn(train_X, train_y, test_X, [k]).reshape(n_data_all)
    #result = (preds_all == test_y)
    #n_corrects = result.sum(axis=0)

    for label in labels:
        print(f'Label: {label}\t', end='')

        indices = np.where(test_y == label)[-1]
        n_data = len(indices)
        preds = preds_all[indices]

        # make confusion matrix
        for i in labels:
            n = (preds == i).sum()
            confusion_matrix[label, i] = n

        # calc accuracy
        n_correct = confusion_matrix[label, label]
        acc = n_correct / n_data
        print(f'#Data: {n_data}\t#Correct: {n_correct}\tAcc: {acc:.3f}')

        result[label] = {
            'data': n_data,
            'correct': n_correct,
            'accuracy': acc,
            }
    result['confusion_matrix'] = confusion_matrix

    # overall score
    n_crr_all = np.diag(confusion_matrix).sum()
    acc_all = n_crr_all / n_data_all
    result['all'] = {
        'data': n_data_all,
        'correct': n_crr_all,
        'accuracy': acc_all,
        }
    print(f'All\t#Data: {n_data_all}\t#Correct: {n_crr_all}\tAcc: {acc_all:.3f}')
    print()
    print('Confusion Matrix:\n', confusion_matrix)
    print()
    return result


def print_result_in_TeX_tabular_format(result):
    labels = list(range(10))
    print('Scores')
    for label in labels:
        print('{} & {} & {} & {:.3f} \\\\'.format(
            label,
            int(result[label]['data']),
            int(result[label]['correct']),
            result[label]['accuracy']
            ))
    print()
    print('Confusion Matrix')
    for i in labels:
        print('{}    '.format(i), end='')
        for j in labels:
            print(' & {}'.format(int(result['confusion_matrix'][i, j])), end='')
        print(' \\\\')
    return


def main():
    # settings
    k_list = list(range(1, 11, 1))
    np.random.seed(0)
    print('Settings')
    print(f'k Candidates: {k_list}\n')

    # load data
    train_X, train_y, test_X, test_y, labels = load_data(
        n_label=None, n_train=None, n_test=None,
        )
    _train_X, _train_y = shuffle(train_X, train_y)
    _train_X, _train_y = split(_train_X, _train_y, n=10)

    # train
    print('Train')
    n_corrects = train(_train_X, _train_y, k_list, save_memory=True)
    print(f'#Correct: {n_corrects}')
    k_best = np.argmax(n_corrects) + 1
    print(f'Best k: {k_best}\n')

    # test
    result = test(train_X, train_y, test_X, test_y, k_best, labels)
    print_result_in_TeX_tabular_format(result)


if __name__ == '__main__':
    main()
