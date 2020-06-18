#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:12:22 2020

@author: jmamath
"""

import numpy as np

# This function is used for all active learning workflows
def sample_random(n_sample, data):
    """
    Sample n_sample from data using a uniform distribution.
    Delete the samples from original data. This function is used to
    provide new data of the active learning procedure from the pool.
    Args:
        n_sample: Int. Number of example to sample
        data: Pool of data.
    Return:
        new_data: the n_sample from data
        data: data - new_data
    """
    len_data = data.shape[0]
    # Sampling from uniform distribution
    indices = np.random.uniform(0,len_data,size = n_sample).astype(np.int16)
    new_data = data[indices,:]
    data = np.delete(data,indices, axis=0)
    return new_data, data

# This function is used to plot the decision boundary that a model learnt
# on a binary classification task
def plot_decision_boundary(weight):
    """
    Get the parameters of the last weight and derive the equation of
    the line it draws.
    """
    w_,b_ = weight[-1]    
    line_x = np.linspace(-1,1,100)
    a = -w_[0]/w_[1]
    b = -b_ / w_[1]    
    line_y = a*line_x + b
    return line_x, line_y

def entropy(p):
    '''
    computes the entropy of one binary probability distribution
    '''
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))


def index_high_values(data, n_sample):
    """    
    Get the index of the n_sample highest value in data
    Args:
        data: np array of unordered value
        n_sample: Int. Number of element to draw from the ordered data
    Return:
        idx_high_value: Indexes of the highest values in data
    """
    if not isinstance(data, (np.ndarray,)):
        data = np.array(data)    
    idx_sorted = np.argsort(data)
    idx_high_value = idx_sorted[-n_sample:]
    return idx_high_value
    
def test_index_high_values():
    # Small test to understand what the program does    
    a = np.random.uniform(0,20,10).astype(np.int)
    id_a = index_high_values(a, 5) 
    return a, id_a
        
def sample_highest_entropy(n_sample, model, data):
    '''
    Sample n_sample from data using a uncertainty sampling.
    We select n_sample with the highest entropy that we call: data_with_high_entropy
    Delete the samples from original data.    
    '''
    entropies = entropy(model.predict(data[:,:2])).squeeze()
    id_high_entropies = index_high_values(entropies, n_sample)
    data_with_high_entropy = data[id_high_entropies]
    data = np.delete(data, id_high_entropies, axis=0)
    return data_with_high_entropy, data

def  sample_lowest_margin(n_sample, model, data):
    """
    Disclaimer: this function works only in binary classification.
    First the function computes the margin between two classes
    Then it select n_sample items with the lowest margin.
    Those items are typically those that the model struggle to classify
    because the margin between the predictions is low.
    """
    pred = model.predict(data[:, :2]) # predict unlabeled data     
    pred_others = 1-pred         # compute  the proabilities of the second class  
    margin = np.abs(pred-pred_others) # compute the difference and take the max 
    index = margin.argsort(axis=0)                 #  get the indexes to sort by margin
    data_sorted = np.take_along_axis(data, index , axis=0)  # the new data sorting 
    data_add_training = data_sorted[:n_sample]  # take the first 10 rows, with the lowest margin       
    data_labelled = data_sorted[n_sample:]     # data without the last 10 rows         
    return data_add_training, data_labelled


def sample_least_confidence(n_sample, model, data):
    """
    This function can be used even for multiclass classification problem.
    It selects the items where the prediction is the least confident
    by sorting 1-prediction.
    """    
    pred = model.predict(data[:, :2])
    least_confidence = 1-pred
    ind = least_confidence.argsort(axis=0)
    data_sorted = np.take_along_axis(data, ind , axis=0)  # the new data sorting     
    data_add_training = data_sorted[-n_sample:]  # take the last 10 rows, with the least confidence        
    data_labelled = data_sorted[:-n_sample]     # data without the last 10 rows 
    return data_add_training, data_labelled
