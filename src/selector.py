#!/usr/bin/env python
# encoding: utf-8
"""
fcbf.py
Created by Prashant Shiralkar on 2015-02-06.
Fast Correlation-Based Filter (FCBF) algorithm as described in 
Feature Selection for High-Dimensional Data: A Fast Correlation-Based
Filter Solution. Yu & Liu (ICML 2003)
"""

import sys
import os
import argparse
import numpy as np

def entropy(vec, base=2):
	" Returns the empirical entropy H(X) in the input vector."
	_, vec = np.unique(vec, return_counts=True)
	prob_vec = np.array(vec/float(sum(vec)))
	if base == 2:
		logfn = np.log2
	elif base == 10:
		logfn = np.log10
	else:
		logfn = np.log
	return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
	"Returns H(X|Y)."
	uy, uyc = np.unique(y, return_counts=True)
	prob_uyc = uyc/float(sum(uyc))
	cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
	return prob_uyc.dot(cond_entropy_x)
	
def mutual_information(x, y):
	" Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
	return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
	" Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
	return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))

def getFirstElement(d):
	"""
	Returns tuple corresponding to first 'unconsidered' feature
	
	Parameters:
	----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	
	Returns:
	-------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	
	t = np.where(d[:,2]>0)[0]
	if len(t):
		return d[t[0],0], d[t[0],1], t[0]
	return None, None, None

def getNextElement(d, idx):
	"""
	Returns tuple corresponding to the next 'unconsidered' feature.
	
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature whose next element is required.
		
	Returns:
	--------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	t = np.where(d[:,2]>0)[0]
	t = t[t > idx]
	if len(t):
		return d[t[0],0], d[t[0],1], t[0]
	return None, None, None
	
def removeElement(d, idx):
	"""
	Returns data with requested feature removed.
	
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature which needs to be removed.
		
	Returns:
	--------
	d : ndarray
		Same as input, except with specific feature removed.
	"""
	d[idx,2] = 0
	return d

def c_correlation(X, y):
	"""
	Returns SU values between each feature and class.
	
	Parameters:
	-----------
	X : 2-D ndarray
		Feature matrix.
	y : ndarray
		Class label vector
		
	Returns:
	--------
	su : ndarray
		Symmetric Uncertainty (SU) values for each feature.
	"""
	su = np.zeros(X.shape[1])
	for i in np.arange(X.shape[1]):
		su[i] = symmetrical_uncertainty(X[:,i], y)
	return su

def su(X, i, j):
	return symmetrical_uncertainty(X[:,i], X[:,j])

def f_correlation(X):
	su = {}
	for i in np.arange(X.shape[1]):
		su[i] = {}
	for i in np.arange(X.shape[1]):
		for j in np.arange(X.shape[1]):
			su[i][j] = symmetrical_uncertainty(X[:,i], X[:,j])
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
			i+= 1
		need_more = k - len(selected)
		idx = np.argsort(removed_thres)[:need_more]
		for i in range(need_more):
			selected.append(removed[idx[i]])
			# print("ADDED MORE", removed[idx[i]])
		return selected
		

