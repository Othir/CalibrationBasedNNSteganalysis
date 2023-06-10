import numpy as np
import os
import platform
from six.moves import cPickle as pickle

DATA_PATH = './data/Cifar10/cifar-10-batches-py/'


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def load_raw_image():
    # dir = DATA_PATH + 'cifar-10-batches-py/'
    dir = DATA_PATH
    train_x = []

    def _load_batch_raw_cifar10(filename):
        batch = np.load(filename)
        data = batch['data']
        return data

    for filename in os.listdir(dir):
        path = dir + filename
        if filename[:4] == 'data':
            x = _load_batch_raw_cifar10(path)
            train_x.append(x)

    train_x = np.concatenate(train_x, axis=0)
    return train_x


def _load_batch_cifar10(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_cifar10():
    # dir = './data/' + 'cifar-10-batches-py/'
    dir = DATA_PATH
    train_x, train_y = [], []

    for filename in [1,2,3,4,5]:
        path = dir + 'data_batch_{}'.format(filename)
        x, y = _load_batch_cifar10(path)
        train_x.append(x)
        train_y.append(y)

    train_x = np.concatenate(train_x, axis=0)

    return train_x


def load_cifar(label_size=10):
    if label_size == 10:
        return load_cifar10()


if __name__ == '__main__':
    pass
