import numpy as np
import scipy.io as sio
from metric import metric
from sklearn import preprocessing
import math
from sklearn.metrics import f1_score, label_ranking_loss, hamming_loss, coverage_error, average_precision_score

def convert_labels(Y):
    Y = (Y > 0) * 1
    # convert to 0/1 labels

    return Y

def load_data(path, tr_rate=0.8, observe_rate=0.2):
    data = sio.loadmat(path)

    if path.find('nus') >= 0:
        X = data['data'].T
        X = preprocessing.scale(X)
        Y = data['target'].T
        Y_P = data['train_p_target'].T
    else:
        X = data['data']
        X = preprocessing.scale(X)
        # pre-processing
        Y = data['target'].T
        Y_P = data['partial_labels'].T
    print('Data size (X, Y, Y_P): ', X.shape, Y.shape, Y_P.shape)

    data_num = int(X.shape[0])
    perm = np.arange(data_num)
    np.random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    Y_P = Y_P[perm]
    part = int(data_num * tr_rate)
    trX = X[0:part, :]
    trY = Y[0:part, :]
    trpY = Y_P[0:part, :]
    tsX = X[part:data_num, :]
    tsY = Y[part:data_num, :]

    return (trX, trY, trpY, tsX, tsY)


def evaluate(Y_test, scores, threshold=0.5):
    return metric(Y_test, 1 * (scores > threshold), scores)


if __name__ == '__main__':
    pass