#!/usr/bin/env python
"""
Adapts from two different works:

https://github.com/shiralkarprashant/FCBF
fcbf.py
Created by Prashant Shiralkar on 2015-02-06.
Fast Correlation-Based Filter (FCBF) algorithm as described in 
Feature Selection for High-Dimensional Data: A Fast Correlation-Based
Filter Solution. Yu & Liu (ICML 2003)

https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
Non-parametric computation of entropy and mutual-information
Adapted by G Varoquaux for code created by R Brette, itself
from several papers (see in the code).
These computations rely on nearest-neighbor statistics
"""

import sys
import numpy as np
from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors

__all__=['FeatureSelector']

EPS = np.finfo(float).eps


def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))


def symmetrical_uncertainty(x, y):
    " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
    return 2.0*mutual_information((x.reshape(-1,1), y.reshape(-1,1)), k=3)/\
    (entropy(x.reshape(-1,1), k=3) + entropy(y.reshape(-1,1), k=3))


def su(X, i, j):
    return symmetrical_uncertainty(X[:, i], X[:, j])


def f_correlation(X):
    su = {}
    for i in np.arange(X.shape[1]):
        su[i] = {}
    for i in np.arange(X.shape[1]):
        for j in np.arange(X.shape[1]):
            su[i][j] = symmetrical_uncertainty(X[:, i], X[:, j])
    return su


class FeatureSelector(object):
    def __init__(self, X, threshold):
        super(FeatureSelector, self).__init__()
        self.X = X
        self.threshold = threshold
        self.su = np.zeros((self.X.shape[1], self.X.shape[1]))
        # self.k = k

    def select(self, to_select, k):
        selected = []
        removed = []
        removed_thres = []

        i = 0
        while len(selected) < k and i < self.X.shape[1]:
            to_add = True
            current_feat = to_select[i]
            if len(selected) > 0:
                for feat in selected:
                    if self.su[current_feat, feat] == 0:
                        correlation = su(self.X, current_feat, feat)
                        self.su[current_feat, feat] = correlation
                        self.su[feat, current_feat] = correlation
                    correlation = self.su[current_feat, feat]
                    if correlation >= self.threshold:
                        to_add = False
                        removed.append(current_feat)
                        removed_thres.append(correlation)
            if to_add:
                selected.append(current_feat)
            i += 1
        need_more = k - len(selected)
        idx = np.argsort(removed_thres)[:need_more]
        for i in range(need_more):
            selected.append(removed[idx[i]])
        return selected
