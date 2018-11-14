# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:36:04 2017

@author: applu
This method will split the training data

Code reference:
    random.permutation - shuffles numerical values
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.permutation.html
    
    Array handling:
    http://www.i-programmer.info/programming/python/3942-arrays-in-python.html
    
    dataframe.iloc
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html
"""

import numpy as np

def split_train_test(data, test_ratio):
    #shuffle the index of data by randomizing it
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
