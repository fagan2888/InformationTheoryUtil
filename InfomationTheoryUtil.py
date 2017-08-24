import numpy as np
from collections import Counter

def entropy_y(y):
    n = len(y)
    ret = 0
    for i in np.unique(y):
        pi = np.sum(y==i) * 1.0 / n
        ret += - pi * np.log2(pi)
    return ret

def cond_entropy(X, y, feat_index):
    assert len(X) == len(y)
    ret = 0
    unique_feats = np.unique(X[:, feat_index])
    # print('unique feats:', unique_feats)
    for feat in unique_feats:
        # print('feat {} y {}'.format(feat, y[X[:,feat_index]==feat]))
        ret += entropy_y(y[X[:,feat_index]==feat])
    return ret

def info_gain(X, y, feat_index):
    assert len(X) == len(y)
    H_D = entropy_y(y)
    H_D_A = cond_entropy(X, y, feat_index)
    return H_D - H_D_A

def info_gain_ratio(X, y, feat_index):
    assert len(X) == len(y)
    g = info_gain(X, y, feat_index)
    H_A_D = 0
    unique_feats = np.unique(X[:, feat_index])
    for feat in unique_feats:
        pi = np.sum(X[:, feat_index]==feat) * 1.0 / len(X)
        H_A_D += - pi * np.log2(pi)
    return g / H_A_D

def gini_y(y):
    ret = 0
    for k in np.unique(y):
        pi = np.sum(y==k) * 1.0 / len(y)
        ret += pi * (1 - pi)
    return ret

if __name__ == '__main__':
    assert 2.0 == entropy_y(np.array([1, 2, 3, 4]))
    assert 1.0 == entropy_y(np.array([1, 2]))
    assert 1.0 == entropy_y(np.array([1, 2, 1, 2]))
    assert 0.0 == entropy_y(np.array([1, 1, 1, 1]))

    X = np.array([[1, 1], [2, 2], [2, 2], [3, 3], [3, 4], [3, 5], [4, 1], [4, 1], [4, 4], [4, 4]])
    y = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

    print('entropy_(y):{}'.format(entropy_y(y)))
    print('cond_entropy:{}'.format(cond_entropy(X, y, feat_index=0)))
    print('info_gain:{}'.format(info_gain(X, y, feat_index=0)))
    print('info_gain_ratio:{}'.format(info_gain_ratio(X, y, feat_index=0)))
    print('gini:{}'.format(gini_y([1, 1, 1])))
    print('gini:{}'.format(gini_y([1, 2])))
    print('gini:{}'.format(gini_y([1, 2, 1, 2])))
